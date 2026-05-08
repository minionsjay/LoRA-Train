import asyncio
import logging
import random
from datetime import datetime

from .models import CountryTaxonomy, LocalViolation, AppConfig, GenerationPlan, GenerationStats
from .prompt_builder import PromptBuilder, get_positive_count
from .llm_client import LLMClient, generate_and_parse
from .output_writer import OutputWriter
from .adversarial import generate_adversarial_variants

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, config: AppConfig, llm_client: LLMClient, output_writer: OutputWriter):
        self.config = config
        self.llm = llm_client
        self.writer = output_writer
        self.prompts = PromptBuilder(model=config.llm.model)

    def _build_plan(self, country: CountryTaxonomy, violation: LocalViolation) -> GenerationPlan:
        dt = violation.detection_type
        pos_count = get_positive_count(dt)
        adv_count = pos_count // 2 if dt in ("keyword_sensitive", "hybrid") else pos_count // 10
        boundary_counts = {
            bc.scenario: self.config.generation.samples_per_label.negative.per_boundary_case
            for bc in violation.boundary_cases
        }
        gen_neg_count = self.config.generation.samples_per_label.negative.general_non_violation
        output_path = f"{self.config.output.dir}/{country.country_code}/{violation.label}.jsonl"

        return GenerationPlan(
            country=country,
            violation=violation,
            positive_count=pos_count,
            adversarial_count=adv_count,
            boundary_counts=boundary_counts,
            general_negative_count=gen_neg_count,
            output_path=output_path,
        )

    async def generate_all(self, countries: list[CountryTaxonomy]):
        """Generate data for all countries and their local violations."""
        # Build all plans
        plans: list[GenerationPlan] = []
        for country in countries:
            for violation in country.local_violations:
                if not self.config.generation.force and self.writer.exists(country.country_code, violation.label):
                    logger.info(f"Skipping {violation.label} (output exists, use --force to overwrite)")
                    continue
                plan = self._build_plan(country, violation)
                plans.append(plan)
                logger.info(
                    f"Planned {violation.label}: "
                    f"+{plan.positive_count}pos +{plan.adversarial_count}adv "
                    f"+{sum(plan.boundary_counts.values())}bnd +{plan.general_negative_count}gen"
                )

        if not plans:
            logger.info("No labels to generate (all outputs exist, use --force to overwrite)")
            return

        logger.info(f"Generating {len(plans)} labels across {len({p.country.country_code for p in plans})} countries")

        # Generate sequentially per plan (each plan has its own async batching internally)
        for plan in plans:
            await self._generate_for_plan(plan)

    async def _generate_for_plan(self, plan: GenerationPlan):
        country = plan.country
        violation = plan.violation
        label = violation.label
        cc = country.country_code
        temp = self.config.generation.temperature
        all_samples = []

        stats = GenerationStats(country_code=cc, label=label)

        # --- Phase 1: Positive samples (batched to avoid LLM truncation) ---
        BATCH_SIZE = 50
        pos_batches = [
            min(BATCH_SIZE, plan.positive_count - i * BATCH_SIZE)
            for i in range((plan.positive_count + BATCH_SIZE - 1) // BATCH_SIZE)
        ]
        pos_samples = []
        for batch_idx, batch_count in enumerate(pos_batches):
            logger.info(f"[{label}] Generating positive batch {batch_idx+1}/{len(pos_batches)} ({batch_count} samples)...")
            pos_req = self.prompts.build_positive(country, violation, batch_count)
            batch_samples = await generate_and_parse(self.llm, pos_req, temp)
            for s in batch_samples:
                s.setdefault("language", country.languages[0])
                s.setdefault("adversarial_technique", None)
                s.setdefault("region", country.region)
                s.setdefault("cultural_contexts", violation.cultural_context)
            pos_samples.extend(batch_samples)
            stats.api_calls += 1
            logger.info(f"[{label}] Positive batch {batch_idx+1}: got {len(batch_samples)} samples")
        all_samples.extend(pos_samples)
        stats.positive_samples += len(pos_samples)
        logger.info(f"[{label}] Total positive: {len(pos_samples)} samples across {len(pos_batches)} batches")

        # --- Phase 1b: Adversarial augmentation ---
        if plan.adversarial_count > 0 and pos_samples:
            logger.info(f"[{label}] Generating {plan.adversarial_count} adversarial variants...")
            # Pick random positive samples to augment
            base_for_adv = random.sample(pos_samples, min(len(pos_samples), max(1, plan.adversarial_count // 3)))
            adv_batch_size = max(1, plan.adversarial_count // len(base_for_adv))

            for base_sample in base_for_adv:
                adv_req = self.prompts.build_adversarial(
                    base_sample["text"], country, violation, adv_batch_size
                )
                adv_samples = await generate_and_parse(self.llm, adv_req, temp)
                for s in adv_samples:
                    s.setdefault("language", base_sample.get("language", country.languages[0]))
                    s.setdefault("label", label)
                    s.setdefault("is_violation", True)
                    s.setdefault("severity", violation.severity)
                    s.setdefault("country_code", cc)
                    s.setdefault("region", country.region)
                    s.setdefault("detection_type", violation.detection_type)
                    s.setdefault("cultural_contexts", violation.cultural_context)
                    if "original_text" not in s:
                        s["original_text"] = base_sample["text"]
                all_samples.extend(adv_samples)
                stats.adversarial_samples += len(adv_samples)
                stats.api_calls += 1

            logger.info(f"[{label}] Got {stats.adversarial_samples} adversarial variants")

        # Also generate programmatic adversarial variants for keyword_sensitive labels
        if violation.detection_type in ("keyword_sensitive", "hybrid") and violation.triggers.keywords:
            programmatic_adv = []
            for s in pos_samples[:plan.adversarial_count]:
                variants = generate_adversarial_variants(
                    s["text"],
                    trigger_keywords=violation.triggers.keywords,
                    n_variants=2,
                )
                for v in variants:
                    programmatic_adv.append({
                        "text": v["text"],
                        "label": label,
                        "is_violation": True,
                        "severity": violation.severity,
                        "country_code": cc,
                        "region": country.region,
                        "detection_type": violation.detection_type,
                        "language": s.get("language", country.languages[0]),
                        "cultural_contexts": violation.cultural_context,
                        "adversarial_technique": v["adversarial_technique"],
                        "generation_strategy": "adversarial_augmentation",
                        "model_used": "programmatic",
                        "metadata": {"original_text": s["text"]},
                    })
            all_samples.extend(programmatic_adv)
            stats.adversarial_samples += len(programmatic_adv)

        # --- Phase 2: Negative samples from boundary cases ---
        for boundary in violation.boundary_cases:
            bc_count = plan.boundary_counts.get(boundary.scenario, 5)
            logger.info(f"[{label}] Generating {bc_count} boundary negative samples...")
            neg_req = self.prompts.build_negative_boundary(country, violation, boundary, bc_count)
            neg_samples = await generate_and_parse(self.llm, neg_req, temp * 0.85)
            for s in neg_samples:
                s.setdefault("language", country.languages[0])
                s.setdefault("label", label)
                s.setdefault("is_violation", False)
                s.setdefault("country_code", cc)
                s.setdefault("region", country.region)
                s.setdefault("detection_type", violation.detection_type)
                s.setdefault("severity", violation.severity)
                s.setdefault("cultural_contexts", violation.cultural_context)
                s.setdefault("generation_strategy", "boundary_negative")
                s.setdefault("boundary_scenario", boundary.scenario)
            all_samples.extend(neg_samples)
            stats.api_calls += 1

        # --- Phase 3: General negative samples ---
        if plan.general_negative_count > 0:
            logger.info(f"[{label}] Generating {plan.general_negative_count} general negative samples...")
            gen_req = self.prompts.build_general_negative(country, violation, plan.general_negative_count)
            gen_samples = await generate_and_parse(self.llm, gen_req, temp * 0.7)
            for s in gen_samples:
                s.setdefault("language", country.languages[0])
                s.setdefault("label", label)
                s.setdefault("is_violation", False)
                s.setdefault("country_code", cc)
                s.setdefault("region", country.region)
                s.setdefault("detection_type", violation.detection_type)
                s.setdefault("severity", violation.severity)
                s.setdefault("cultural_contexts", [])
                s.setdefault("generation_strategy", "general_negative")
            all_samples.extend(gen_samples)
            stats.api_calls += 1

        # --- Write all samples ---
        self.writer.write_samples(all_samples, cc, label)
        stats.total_samples = len(all_samples)
        stats.completed_at = datetime.now().isoformat()
        self.writer.write_stats(cc, label, stats.model_dump())

        logger.info(
            f"[{label}] Complete: {stats.total_samples} total "
            f"({stats.positive_samples}+, {stats.adversarial_samples}adv, "
            f"{stats.total_samples - stats.positive_samples - stats.adversarial_samples}-)"
        )
