# End-to-End Agentic System for Medical Imaging Appropriateness

The number of unnecessary imaging procedures is increasing, harming patients and straining healthcare systems. Although the ACR Appropriateness Criteria offer evidence-based guidance on selecting appropriate imaging, they remain underutilized in clinical workflows. 
With the growing capabilities of LLM-based reasoning, there is now an opportunity to bridge this gap by enabling more trustworthy and transparent imaging referrals. This study introduces an LLM-based Reasoning Agent trained via Reinforcement Learning (RL), specifically using Group Relative Policy Optimization (GRPO), to replicate expert clinical reasoning and recommend appropriate imaging; marking the first application of RL leveraging structured reasoning from the ACR Criteria. It is also the first to systematically compare reasoning-focused reward functions and evidence integration strategies in medicine, placing reasoning quality at the core to build clinician trust and enable real-world deployment. Our best lightweight RL model, MedReason-Embed, outperforms the baseline by 18\% in macro F1, achieves significantly higher reasoning capabilities, and surpasses both larger models and those trained with alternative strategies, showing that reasoning-aligned supervision enables efficient, trustworthy clinical AI. To that end, we also develop a modular end-to-end agentic system that replicates the full imaging referral process, incorporating PubMed-based evidence retrieval and generating well-justified recommendations. The system aims to generalize beyond static guidelines and operates fully autonomously, with potential for continuous updates. This work highlights the promise of reasoning-focused RL within full-system architectures to enable autonomous, trustworthy, and explainable clinical decision-making in radiology.

### 🧠 Agent Architecture Overview

<img src="/figures/final_architecture.png" alt="Agent Architecture" width="800"/>


The system is composed of specialized agents working together in a modular pipeline:

- **ICD Mapping Agent**  
  Maps noisy Italian clinical notes to standardized **ICD-9-CM** codes using LLM-based normalization and embedding retrieval. This is the first step in standardizing input.

- **ACR Criteria Checker**  
  Checks if the mapped ICD-9-CM code matches any condition in the **ACR Appropriateness Criteria**.  
  - If it **does**, the pipeline proceeds directly to the **Reasoning Agent** and uses the ACR medical evidence.
  - If it **does not**, the pipeline triggers the **Medical Review** and **Post-Filtering Agents** to gather and curate relevant evidence.

- **Medical Review Agent**  
  When no ACR variant exists for the diagnosis, this agent uses **DeepRetrieval** to search PubMed for relevant studies describing imaging guidance for the clinical condition.

- **Post-Filtering Agent**  
 Applies a lightweight ML-based quality filter to the retrieved literature, using features like design (e.g., RCTs, cohort studies) and sample size, and assigning strength of evidence according to the GRADE scale.

- **Reasoning Agent**  
  The core agent, trained using **Group Relative Policy Optimization (GRPO)** to replicate **stepwise expert reasoning traces** from the ACR criteria.  
  - Supports multiple reward functions (e.g., format, answer correctness, reasoning alignment)
  - Can incorporate external medical evidence to produce more accurate and transparent justifications.

## ⚙️ Installation and Setup
- Requires **Python 3.10**.  
- Install dependencies from the `requirements.txt` file.  
- Detailed implementation steps are included in the README files inside each script directory.

## 📊 Results
Key experimental results are stored in the `results` directory, organized into files for:
- **Generalization**
- **ICD Coding**
- **Literature Retrieval**
- **Reasoning Agent**

The full document is also given under `final_document.pdf`.


## 🚀 Personal Note

I'm especially proud of this Dissertation project. I made it a goal to contribute to it **every single day** for 6 weeks-  a commitment you can see in the GitHub commit heatmap below!  

<img src="/figures/personal_contributions.png" alt="Contributions" width="200"/>

## 🤝 Open to Collaboration
If you're working on similar topics in medical AI or clinical reasoning systems, feel free to reach out! I am currently working on advancing the system and I would love to connect, exchange ideas, or collaborate on future work. 🧠📚💬