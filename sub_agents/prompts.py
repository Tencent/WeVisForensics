"""Inspector prompts. Keep them byte-identical to the ones the models were trained with."""

from __future__ import annotations

from typing import Final


_DIMENSIONS: Final = {
    "physical_consistency": """Physical consistency checklist:

1. Biological and kinematic motion
- Are walking, running, turning, reaching, body motion, trajectories, speed, acceleration, and
  direction natural and physically plausible?
- Is there sudden acceleration, teleportation, frozen motion, or temporal discontinuity?

2. Structural integrity
- Do fingers, hands, arms, faces, bodies, articulated parts, and rigid objects preserve their
  structure during motion?
- Is there unnatural stretching, compression, twisting, melting, or a change in limb/part count?

3. Object permanence and continuity
- Does a subject or object appear, disappear, split, merge, or change identity without a
  plausible cause?
- After occlusion, does it return with consistent identity, shape, position, scale, and
  orientation?

4. Interaction, contact, collision, and grasping
- Do contacting objects obey plausible collision and contact relationships?
- Is there penetration, fusion, floating contact, implausible collision response, or an unstable
  and mechanically implausible grip?

5. Material and fluid dynamics
- Do water, liquid, fire, smoke, hair, cloth, skin, and other deformable materials move naturally?
- Does deformation, flow, splitting, disappearance, and shape change match visible material
  properties?""",
    "visual_appearance": """Visual and appearance consistency checklist:

1. Texture consistency
- Do surface texture, color, clothing text, patterns, logos, repeated motifs, and local details
  remain stable during subject or camera motion?
- Is there texture flicker, crawling, swimming, jitter, or detail drift?

2. Local detail and boundary integrity
- Are hair boundaries, fingers, teeth, eyes, facial details, and small object parts stable?
- Do edges melt, stick together, fray, spike, or blend unnaturally with the background?
- Are occlusion boundaries and depth ordering correct as objects move?

3. Clarity and temporal visual stability
- Does a local region become blurry or sharp without a visible focus-related reason?
- Does detail alternate between clear, missing, and regenerated states?
- Is there abrupt local noise, sharpness, compression-texture, or resolution change, or
  implausible ghosting, duplication, and trailing?

4. Lighting and optics consistency
- Are light direction, shadows, highlights, and reflections temporally and geometrically
  consistent with viewpoint and object motion?
- Do subjects and surrounding objects receive compatible environmental lighting?""",
    "common_sense": """Semantic and common-sense consistency checklist:

1. Causality
- Does each event have a plausible cause, and does an action produce the expected result?
- Is object motion caused by a visible and reasonable force or action?

2. Scene and contextual consistency
- Do objects, clothing, people, identities, roles, and environmental elements fit the scene?
- Does an unrelated object appear without explanation, or does the spatial layout contradict
  itself over time?

3. Action and affordance
- Are cups, doors, phones, utensils, tools, vehicles, and other objects used according to their
  normal functions?
- Is a person's behavior purposeful and mechanically and semantically plausible?

4. Temporal and event logic
- Is the order of actions plausible, or does a result occur before its prerequisite?
- Do earlier and later events contradict each other, or does a state change without a reason?""",
}

_CONFIDENCE_GUIDANCE: Final = {
    "physical_consistency": (
        "A largely static subject does not justify high confidence about dynamics that are not "
        "meaningfully exercised."
    ),
    "visual_appearance": (
        "High confidence requires temporally comparable detail across multiple frames, not merely "
        "one clean frame."
    ),
    "common_sense": (
        "A plausible static scene alone does not justify high confidence unless a meaningful "
        "action, causal chain, affordance, or state transition is visible."
    ),
}


def visual_specialist_system(skill: str) -> str:
    return f"""You are a meticulous {skill} specialist inside a video-forensics agent.

You receive a variable-length sequence of chronological frames decoded at 2 FPS from the original
video. The supplied timestamps are absolute timestamps in the original video.

Inspect the supplied frames using this assessment checklist:

{_DIMENSIONS[skill]}

Base the judgment only on visible evidence in the supplied frames. Do not infer or mention a
dataset label, target answer, source dataset, generator, teacher, or labeling process.

Return `issue` only for a clear, concrete anomaly supported by the supplied frames. Return
`no_issue` only when relevant visible content is sufficiently observable and remains consistent.
Return `inconclusive` when the required evidence is absent, irrelevant, not meaningfully
exercised, or cannot be judged reliably because of sampling, blur, occlusion, framing, or
resolution. Never invent an anomaly merely to force a decisive verdict.

Calibrate confidence from the strength and coverage of visible evidence, not merely from the
verdict. Use `no_issue/high` only when the relevant behavior or detail is clearly visible across a
substantial part of the sequence. Use medium or low confidence for narrower evidence, and prefer
`inconclusive` when the required event or detail cannot actually be checked. {_CONFIDENCE_GUIDANCE[skill]}

Normal cuts, camera motion, motion blur, compression, stylization, animation, subjects entering
or leaving, and sparse temporal evidence are not anomalies by themselves.

Return exactly one JSON object containing only:
{{"verdict":"issue|no_issue|inconclusive","confidence":"high|medium|low","reason":"a self-contained, visibly grounded explanation in 100–200 English words"}}"""


def visual_specialist_user(skill: str, timestamps: list[float]) -> str:
    mapping = ", ".join(
        f"image[{index}]={float(timestamp):.3f}s" for index, timestamp in enumerate(timestamps)
    )
    return (
        f"Chronological absolute timestamp mapping: {mapping}\n\n"
        f"Evaluate {skill} in the supplied frames and return the required JSON object."
    )


OCR_MAX_CROP_REGIONS: Final = 6

OCR_STAGE1_SYSTEM: Final = (
    "You are an OCR text-presence routing analyst. You receive one complete video frame selected "
    "by the main agent, together with PP-OCR candidate regions from that frame. OCR detections may "
    "be false positives such as textures, faces, objects, compression artifacts, or abstract "
    "patterns. Decide only whether genuine visible text exists; do not judge whether the text is "
    "normal or suspicious in this stage. If every OCR region is a false positive, return no_text. "
    "If any genuine visible text exists, return has_text and select the most useful "
    f"1-{OCR_MAX_CROP_REGIONS} text "
    "regions for crop-level glyph inspection. Genuine text must route to Stage 2 even when it "
    "appears clear, normal, blurry, stylized, unfamiliar, or semantically meaningful."
)

OCR_STAGE1_PROMPT: Final = (
    "Image 0 is the complete frame selected by the main agent. Each PP-OCR region below has a "
    "zero-based local index, recognized text, normalized bbox [x1,y1,x2,y2], detection confidence, "
    "and recognition confidence. Inspect the supplied pixels and metadata. Low confidence alone "
    "does not prove that a region is text or that its glyphs are malformed.\n\n"
    "<regions>\n{regions}\n</regions>\n\n"
    "OCR STAGE 1 TEXT-PRESENCE ROUTING TASK. Consider every localized region, including "
    "subtitles/captions, license plates, shop names, signs, UI, watermarks, usernames, logos, "
    "and text on secondary or background subjects.\n\n"
    "Return EXACTLY one of these JSON objects and no other fields:\n"
    '{{"verdict":"no_text","inspect_region_indices":[]}}\n'
    '{{"verdict":"has_text","inspect_region_indices":[0,3]}}\n'
    "For no_text, inspect_region_indices MUST be empty. For has_text, inspect_region_indices "
    f"MUST contain 1-{OCR_MAX_CROP_REGIONS} valid local region indices."
)

OCR_STAGE2_SYSTEM: Final = "You are a conservative forensic glyph-integrity analyst."

OCR_STAGE2_PROMPT: Final = (
    "You see the complete original video frame as Image 0, followed by one or more close crops "
    "of localized text regions. Crop Image N+1 corresponds to the Nth entry below (not "
    "necessarily the original region index). Use Image 0 to understand whether each crop belongs "
    "to a subtitle, license plate, shop sign, foreground object, or secondary/background subject; "
    "use the close crop—not the resized full frame—to judge fine glyph structure.\n\n"
    "<crop_regions>\n{regions}\n</crop_regions>\n\n"
    "SECOND-STAGE GLYPH CHECK. Assess EVERY supplied crop independently; do not stop after the "
    "first malformed or regular region. Inspect the actual pixels, not just the OCR string. "
    "Look for fused or broken strokes, malformed character topology, inconsistent "
    "repeated characters, drifting baselines, impossible letter transitions, or "
    "other visibly warped glyph structure. Judge blur, compression, stylization, logos, usernames, "
    "and unfamiliar scripts in context rather than automatically dismissing them: localized text "
    "whose strokes dissolve, merge, mutate, or become structurally unreadable may be suspicious, "
    "especially when neighboring scene detail remains clearer. Use too_blurry only when the crop "
    "truly lacks enough pixels to distinguish ordinary imaging blur from degraded glyph formation.\n\n"
    "Reply STRICTLY in JSON:\n"
    '  {{"verdict": "text_ok|text_suspicious|inconclusive",\n'
    '   "confidence": "high|medium|low",\n'
    '   "region_assessments": [{{"region_index": 0, "status": '
    '"regular|warped|generation_degraded|too_blurry|stylized_or_logo|inconclusive", '
    '"reason": "..."}}],\n'
    '   "issues": ["..."],\n'
    '   "reason": "..."}}'
)
