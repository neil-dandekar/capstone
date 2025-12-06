Below is a **clean, self-contained report** you can directly reuse (capstone write-up, appendix, or internal doc). It’s written to be technically precise but readable by an ML-savvy audience.

# Concept Bottleneck Large Language Models and Concept Intervention

## 1. Concept Bottleneck Large Language Models (CB-LLMs)

### 1.1 Motivation

Large language models (LLMs) are powerful but opaque. While they achieve strong performance on classification and generation tasks, it is often unclear _why_ a particular output was produced or _which internal features_ were responsible. Traditional post-hoc interpretability methods (e.g., attention visualization, probing, or saliency) provide limited causal insight.

**Concept Bottleneck Large Language Models (CB-LLMs)** address this by introducing an explicit, interpretable intermediate representation composed of **human-named concepts**. This bottleneck enables both faithful explanations and direct intervention.

---

### 1.2 Model Architecture

A CB-LLM modifies a standard LLM by inserting a **concept bottleneck layer** between the encoder and the output layer.

The architecture can be described as:

1. **Encoder**
   A pretrained LLM processes the input text ( x ) and produces a hidden representation:
   [
   h(x) = f_{\text{enc}}(x)
   ]

2. **Concept Layer (Bottleneck)**
   A learned linear or shallow mapping projects the hidden representation into a fixed set of ( K ) concepts:
   [
   A(x) \in \mathbb{R}^K
   ]
   Each dimension ( A*j(x) ) corresponds to a specific, human-interpretable concept (e.g., \_Positive Sentiment*, _Overpriced_, _Politics_, _Toxicity_).

    To ensure interpretability and sparsity, activations are typically passed through a ReLU:
    [
    A^+(x) = \max(0, A(x))
    ]

3. **Linear Output Layer**
   The final prediction is computed as a linear function of concept activations:
    - **Classification**:
      [
      \text{logit}*c = \sum*{j=1}^K W_{c,j} \cdot A^+_j(x)
      ]
    - **Generation** (per decoding step):
      [
      \text{logits}_t = W \cdot A^+_t
      ]

Because the mapping from concepts to outputs is linear, **each concept’s contribution to the output is explicit and computable**.

---

### 1.3 Why the Bottleneck Matters

The concept bottleneck provides three key properties:

1. **Interpretability**
   Each decision is decomposable into concept-level contributions.

2. **Faithfulness**
   Because the final layer is linear, explanations are not post-hoc approximations. A concept’s contribution is exactly:
   [
   W_{c,j} \cdot A^+_j(x)
   ]

3. **Intervenability**
   Concepts can be directly modified to test counterfactuals or steer model behavior.

---

## 2. Concept Intervention: Core Idea

### 2.1 Definition

**Concept intervention** is the act of **manually modifying the concept bottleneck representation or its downstream influence** and observing how the model’s output changes.

Instead of editing the input text, intervention operates on the internal concept space:
[
A^+(x) ;\longrightarrow; \tilde{A}(x)
]

The modified concept representation is then used to compute the output:
[
\text{output} = W \cdot \tilde{A}(x)
]

This enables controlled, causal experiments such as:

- “What if the _Overpriced_ concept were suppressed?”
- “What if _Positive Sentiment_ were stronger?”
- “What happens if this concept is completely removed from decision-making?”

---

## 3. Types of Concept Intervention

The paper distinguishes **two fundamentally different intervention mechanisms**.

---

### 3.1 Activation (Representation-Level) Intervention

#### Description

Activation intervention modifies the **value of concept activations for a specific input**.

Given baseline activations:
[
A_j(x)
]

We define an intervened activation:
[
\tilde{A}_j(x) = g(A_j(x))
]

Common intervention operators include:

1. **Override**
   [
   \tilde{A}_j(x) = v
   ]
   Forces a concept to take a fixed value.

2. **Additive**
   [
   \tilde{A}_j(x) = A_j(x) + \Delta
   ]
   Increases or decreases the concept strength.

3. **Scaling**
   [
   \tilde{A}_j(x) = s \cdot A_j(x)
   ]
   Amplifies or suppresses the concept proportionally.

After intervention, ReLU is applied if required:
[
\tilde{A}^+_j(x) = \max(0, \tilde{A}_j(x))
]

---

#### Purpose

Activation intervention is used for:

- Counterfactual analysis
- Measuring causal influence of concepts
- Steering generation behavior
- Stress-testing model reliance on specific concepts

This is the **primary intervention mechanism** used for interpretability experiments.

---

### 3.2 Weight (Decision-Level) Intervention (Concept Unlearning)

#### Description

Weight intervention modifies how much a concept influences the output, **independent of the input**.

Instead of changing ( A(x) ), we change the output weights ( W ):

- **Hard unlearning**:
  [
  W_{:,j} = 0
  ]
  The concept contributes nothing to any output.

- **Soft unlearning**:
  [
  W_{:,j} = \alpha \cdot W_{:,j}, \quad 0 < \alpha < 1
  ]
  The concept’s influence is reduced.

---

#### Purpose

Weight intervention is used for:

- Removing biased or sensitive concepts
- Studying robustness after concept removal
- Structural edits to the decision logic

Unlike activation intervention, this does **not depend on the input**.

---

## 4. Concept Intervention in Classification

### 4.1 Process

For a classification task:

1. Encode input text ( x )
2. Compute concept activations ( A(x) )
3. Apply intervention → ( \tilde{A}(x) )
4. Compute logits:
   [
   \text{logit}*c = \sum_j W*{c,j} \cdot \tilde{A}^+_j(x)
   ]
5. Compare baseline vs intervened predictions

---

### 4.2 Explanation and Faithfulness

Because the model is linear, the contribution of each concept to the predicted class is:
[
\text{contribution}*j = W*{\hat{c},j} \cdot A^+_j(x)
]

This allows:

- Exact ranking of influential concepts
- Predictable effects of intervention
- Faithful explanations tied directly to model computation

Classification intervention is:

- Single-step
- Input-specific
- Easy to explain and visualize

---

## 5. Concept Intervention in Generation

Generation introduces temporal complexity.

### 5.1 Generation Model Structure

At each decoding step ( t ):

1. Compute hidden state ( h_t )
2. Compute concept activations ( A_t )
3. Compute token logits:
   [
   \text{logits}_t = W \cdot A^+_t
   ]
4. Sample next token

---

### 5.2 Static Concept Intervention (Most Common)

The paper primarily uses **static intervention** for generation:

- Define a target concept vector ( \tilde{A} )
- Replace ( A_t ) with ( \tilde{A} ) at **every decoding step**

Example:

- Set target concept to a large value
- Set all other concepts to zero

This enforces a global constraint on generation and enables:

- Promptless generation
- Topic and style steering
- High steerability scores

---

### 5.3 Dynamic (Token-Level) Observation

The paper also analyzes concept activations over time:

- Different concepts activate as different tokens are processed
- This is often observational rather than intervened

In principle, interventions could be applied dynamically per step, but static intervention is simpler and more stable.

---

### 5.4 Why Generation Intervention Is Stronger

Compared to classification:

- Intervention affects **every token**
- Effects compound over time
- Concepts act as global constraints on decoding

As a result, small interventions can lead to large semantic changes.

---

## 6. Summary of Intervention Mechanisms

| Mechanism               | Acts On                      | Input-Dependent | Typical Use               |
| ----------------------- | ---------------------------- | --------------- | ------------------------- |
| Activation intervention | Concept activations ( A(x) ) | Yes             | Counterfactuals, steering |
| Weight intervention     | Output weights ( W )         | No              | Unlearning, bias removal  |
| Generation repetition   | ( A_t ) at each step         | Yes             | Strong steering           |

---

## 7. Implications for Interpretability Tools

A CB-LLM interpretability tool should:

- Explicitly distinguish **activation vs weight interventions**
- Treat classification and generation differently
- Support baseline vs intervened comparisons
- Surface concept contributions as first-class evidence
- Enable reproducibility via saved intervention configurations
