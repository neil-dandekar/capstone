# CB-LLM Concept Intervention GUI — Frontend Redesign (Codex Prompt)

## Goal

Redesign the current frontend layout into a clear, researcher-friendly interface for **concept intervention** in a **Concept Bottleneck LLM (CB-LLM)**. The current layout is confusing. Reorganize the UI into **three vertical sections** (top → bottom), with a clean information hierarchy, strong affordances for experimentation, and reproducibility.

This tool supports two tasks:

1. **Text classification** (user supplies text, model outputs label + confidence)
2. **Text generation** (user supplies prompt, model generates text)

The core workflow:

1. Choose dataset + task
2. Inspect and modify **concept neurons** (activations and/or downstream weights)
3. Run baseline and run intervened
4. Compare outputs and inspect **evidence** (top concepts, contributions, etc.)
5. Save/load intervention “recipes” for reproducibility

---

## Non-negotiable Layout Structure (3 Sections)

### Section 1 (Top): Concept Bottleneck Control Panel

Purpose: show concept neurons (“concept bottleneck”), allow editing interventions, and provide experiment controls.

**Layout:**

- Full-width top section with a two-column interior:
    - **Left column (Concept Browser)**: list + search + sorting
    - **Right column (Intervention Editor)**: edit interventions + presets + save/load

**Required components:**

1. **Model/Dataset/Task Header**
    - Dropdowns or selectors for:
        - `Task Mode`: `Classification` | `Generation`
        - `Dataset`: list of available datasets + optional “Custom”
        - (Optional) `Model Checkpoint` if multiple exist
    - A small status line: “Loaded: <model>, <dataset>, <task>”

2. **Concept Browser (Left)**
    - Search input: “Search concepts…”
    - Sort options:
        - `By contribution` (enabled after a run)
        - `By activation` (enabled after a run)
        - `Alphabetical`
    - Concept list rows show at minimum:
        - Concept name
        - Activation (baseline, when available)
        - Activation (intervened, when available)
        - A small indicator if this concept has active interventions (badge)
    - Row click selects a concept; multi-select should be supported for batch edits (Shift/Ctrl).
    - (Optional) Expandable row details:
        - concept id
        - summary stats or notes

3. **Intervention Editor (Right)**
   Must support **two intervention types**:

    **A) Activation intervention (inference-time)**
    - For selected concept(s), allow:
        - Operator: `additive (Δ)` | `override (v)` | `scale (s)`
        - Numeric input / slider for value
    - Show how it will be applied:
        - Example: `A'_j = A_j + Δ` or `A'_j = v` or `A'_j = s * A_j`
    - Toggle: `Apply ReLU after intervention` (if your model uses nonnegative bottleneck behavior)
    - Provide buttons:
        - `Apply to selected`
        - `Clear for selected`
        - `Clear all activation interventions`

    **B) Weight intervention (unlearning / editing downstream linear layer)**
    - Provide per selected concept(s):
        - `Zero outgoing weights` (hard unlearn)
        - `Scale outgoing weights by factor` (soft unlearn)
        - (Optional) per-class editing in classification mode
    - Buttons:
        - `Apply weight intervention`
        - `Clear weight interventions`

4. **Experiment Controls**
    - Buttons:
        - `Run baseline`
        - `Run with intervention`
        - `Compare` (runs both if needed and populates comparison view)
        - `Reset interventions`
    - Toggles:
        - `Lock baseline` (keeps baseline results frozen while user edits interventions)
    - Show “Pending changes” indicator if interventions changed since last run.

5. **Recipes (Reproducibility)**
    - `Save recipe` → exports a JSON object representing current interventions
    - `Load recipe` → imports a JSON recipe
    - Provide a text area/modal showing the JSON with copy button
    - The recipe must include:
        - dataset, task mode, model checkpoint (if applicable)
        - activation interventions: per concept id → operator + value
        - weight interventions: per concept id → action + factor/zero + optional per-class values

---

### Section 2 (Middle): Input Workspace

Purpose: make it easy to run experiments on preloaded examples or user-provided text, for both tasks.

**Layout:**

- A panel with **tabs**:
    - `Classification`
    - `Generation`
- Each tab has dataset-aware example controls and a primary text input.

**Required components:**

1. **Dataset Example Controls**
    - `Example selector` dropdown (preloaded examples)
    - `Random sample` button (pull a random dataset instance)
    - `Use my own text` toggle (switches to a blank input)
    - Display metadata for the loaded example:
        - classification: label (if known)
        - generation: prompt source or id

2. **Classification Tab**
    - Text area: input text to classify
    - Optional field: true label (if user wants evaluation)
    - Button: `Send to model` (or use run buttons above, but must be intuitive)
    - NOTE: The UI must make it obvious that “Run baseline/intervention” uses the current input text.

3. **Generation Tab**
    - Text area: prompt
    - Collapsible “Decoding settings” (advanced)
        - max tokens
        - temperature
        - top_p
        - seed (if supported)
    - Button: `Generate` (or use run buttons above, but must be intuitive)

---

### Section 3 (Bottom): Results & Evidence

Purpose: show baseline vs intervened outputs, plus interpretability evidence (top concepts, contributions) to support research.

**Layout:**

- Default is a **two-column comparison**:
    - Left: Baseline
    - Right: Intervened
- Under each output, show evidence panels.

**Required components:**

1. **Output Comparison**
    - Classification:
        - predicted label + confidence/probability
        - optionally show logits
    - Generation:
        - generated text
        - optional “diff highlight” between baseline and intervened

2. **Evidence: Concepts**
    - Show “Top concepts” (k=10 by default)
    - In classification mode, include a table with:
        - concept name/id
        - activation `A⁺_j(x)` (or equivalent)
        - weight to predicted class `W_pred,j`
        - contribution `W_pred,j * A⁺_j(x)`
    - Allow sorting by contribution and filtering by “only intervened concepts”

3. **Evidence: Token/Time (Generation)**
    - If token-level concept activations exist:
        - show a simple heatmap or timeline view:
            - tokens on x-axis
            - top concepts on y-axis
            - intensity = activation
    - If token-level data is not available, show:
        - top concepts summary for the prompt and/or generation

4. **Reproducibility Footer**
    - Display:
        - dataset
        - task mode
        - model checkpoint/version
        - decoding params (for generation)
        - seed (if supported)
        - intervention recipe JSON (collapsible)

---

## UX Requirements

- The main loop must be obvious: **edit concepts → run → compare → inspect evidence**.
- Advanced settings (per-class weights, decoding params, token heatmaps) should be collapsible to reduce clutter.
- Use clear status indicators:
    - “Baseline locked”
    - “Intervention active: N concepts”
    - “Pending changes”
- Provide fast iteration:
    - concept search must be responsive
    - run buttons must show loading states and disable appropriately

---

## Data/State Requirements (Frontend)

Maintain a centralized state model:

- `selectedTaskMode`: "classification" | "generation"
- `selectedDataset`: string
- `currentInput`: string (classification text or generation prompt)
- `baselineResult`: object | null
- `intervenedResult`: object | null
- `concepts`: list with metadata (name, id, maybe weights/activations)
- `interventions`:
    - `activation`: map concept_id → { op, value, reluAfter }
    - `weights`: map concept_id → { type: "zero" | "scale" | "perClass", value }
- `isBaselineLocked`: boolean
- `pendingChanges`: boolean
- `lastRunConfig`: store dataset/task/input/interventions used for last run

---

## API Integration (Do Not Invent New Backend Logic)

Use the existing API endpoints/hooks already in the repo.

- Refactor the frontend around them.
- If the current code mixes baseline/intervened calls, separate them cleanly:
    - `runBaseline(input, dataset, task)` → baselineResult
    - `runIntervened(input, dataset, task, interventions)` → intervenedResult
- If the backend only has one endpoint, handle baseline by sending empty interventions.

---

## Implementation Instructions for Codex

1. **Reorganize the page layout** into the three sections described above.
2. **Do not break existing functionality**; move components and adjust props/state accordingly.
3. Create/adjust components if needed:
    - `ConceptBrowser`
    - `InterventionEditor`
    - `ExperimentControls`
    - `InputWorkspace` with tabs
    - `ResultsComparison`
    - `EvidencePanel`
    - `RecipeModal`
4. Ensure consistent styling (use the project’s existing styling system).
5. Provide clear empty states:
    - “Run baseline to see activations and contributions”
    - “No intervened results yet”
6. Add small helper tooltips for intervention operators (`add`, `override`, `scale`) and “unlearning”.

---

## Acceptance Criteria

- The UI is visibly segmented into 3 clear sections with headings.
- A user can:
    1. select dataset + task mode
    2. load an example or type their own input
    3. view concept neurons and apply an activation intervention
    4. optionally apply a weight intervention (“unlearn”)
    5. run baseline and intervened
    6. compare results side-by-side
    7. view top concepts and contributions
    8. save/load intervention recipes

---

## Notes / Nice-to-Haves (If Time)

- Add “Preset interventions” dropdown (e.g., “Steer to concept X”).
- Add “Only show intervened concepts” filter in evidence tables.
- Add a small “Intervention summary pill” showing counts and last edited time.

---

## Deliverable

Commit changes that implement the redesigned frontend layout and preserve existing backend calls. Ensure the new UI is logically organized, readable, and optimized for interpretability research workflows.
