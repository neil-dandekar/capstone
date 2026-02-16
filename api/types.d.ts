export type RunMode = "baseline" | "intervened" | "compare";
export type TaskMode = "classification" | "generation";
export type DatasetId =
    | "sst2"
    | "SetFit/sst2"
    | "ag_news"
    | "yelp_polarity"
    | "dbpedia_14";
export type ModelId = "roberta" | "gpt2" | "llama3";

export interface RunContext {
    task_mode: TaskMode;
    dataset_id: DatasetId;
    task_id: "classification" | "generation";
    model_id: ModelId;
    model_checkpoint: string | null;
}

export interface RunInput {
    text: string;
    true_label?: string;
}

export type ActivationOp = "activation" | "weight" | "hybrid";

export interface ActivationIntervention {
    op: "add" | "override" | "scale";
    value: number;
    relu_after?: boolean;
}

export interface WeightIntervention {
    action: "zero" | "scale";
    value?: number;
}

export interface InterventionPayload {
    enabled: boolean;
    mechanism: ActivationOp;
    target_concepts: string[];
    activation?: {
        per_concept: Record<string, ActivationIntervention>;
    };
    weight?: {
        per_concept: Record<string, WeightIntervention>;
    };
    generation_policy?: {
        mode: "static";
        apply_every_step: boolean;
    };
}

export interface GenerationConfig {
    max_tokens: number;
    temperature: number;
    top_p: number;
    seed: number;
}

export interface ResponseOptions {
    include_evidence: boolean;
    include_top_concepts: boolean;
    include_token_time_evidence: boolean;
    top_k_concepts: number;
}

export interface CbllmRunRequest {
    request_id: string;
    run_mode: RunMode;
    context: RunContext;
    input: RunInput;
    intervention: InterventionPayload;
    generation: GenerationConfig;
    response_options: ResponseOptions;
}

export interface TopConcept {
    concept_id: string;
    concept_name: string;
    activation: number;
}

export interface ClassificationOutput {
    predicted_label: string;
    confidence: number;
    probabilities: Record<string, number>;
    logits: Record<string, number> | null;
}

export interface RunResult {
    output: {
        text: string;
        classification: ClassificationOutput | null;
        generation: unknown | null;
    };
    metrics: {
        primary_metric: {
            id: string;
            name: string;
            format: string;
            value: number | null;
        };
        steering_score: number | null;
    };
    evidence: {
        concept_contributions: unknown[];
        top_concepts: TopConcept[];
        token_time: unknown | null;
    };
}

export interface CbllmRunSuccess {
    request_id: string | null;
    status: "ok";
    timing_ms: number;
    effective_config: {
        run_mode: RunMode;
        task_mode: TaskMode;
        dataset_id: string;
        task_id: string;
        model_id: string | null;
        model_checkpoint: string | null;
        intervention_applied: boolean;
    };
    warnings: string[];
    baseline?: RunResult;
    intervened?: RunResult;
}

export interface CbllmRunErrorBody {
    error: {
        code: string;
        message: string;
        details?: unknown[];
    };
    request_id: string | null;
}

