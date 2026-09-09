"""HuggingFace dataset registries for multimodal and V-JEPA fine-tuning."""

from __future__ import annotations

MULTIMODAL_DATASETS: dict[str, str] = {
    "WenxingZhu/multimodal-embedding-100M": "WenxingZhu/multimodal-embedding-100M",
    "SWE-bench/SWE-bench_Multimodal": "SWE-bench/SWE-bench_Multimodal",
    "KIT-MRT/KITScenes-Multimodal": "KIT-MRT/KITScenes-Multimodal",
    "osunlp/Multimodal-Mind2Web": "osunlp/Multimodal-Mind2Web",
    "SamBP069/olives-multimodal-dataset": "SamBP069/olives-multimodal-dataset",
    "multimodal-reasoning-lab/Zebra-CoT": "multimodal-reasoning-lab/Zebra-CoT",
    "ken-sungmin/propagator-multimodal-pretraining-data": "ken-sungmin/propagator-multimodal-pretraining-data",
    "trucmtnguyen/multimodal-product-reviews-lazada": "trucmtnguyen/multimodal-product-reviews-lazada",
    "Perle-ai/multimodal-ct-radiology-reports": "Perle-ai/multimodal-ct-radiology-reports",
    "SilverAvocado/Silver-Multimodal-Dataset": "SilverAvocado/Silver-Multimodal-Dataset",
    "omegalabsinc/omega-multimodal": "omegalabsinc/omega-multimodal",
    "princeton-nlp/SWE-bench_Multimodal": "princeton-nlp/SWE-bench_Multimodal",
    "Y123-wed/Multimodal-Dataset-Image_Text_Table_TimeSeries": "Y123-wed/Multimodal-Dataset-Image_Text_Table_TimeSeries",
    "alibaba-multimodal-industrial-ai/IndustryBench-MIPU": "alibaba-multimodal-industrial-ai/IndustryBench-MIPU",
    "lfsm/multimodal_wiki": "lfsm/multimodal_wiki",
    "tishtakalita/Multimodal-Dataset-Image_Text_Table_TimeSeries": "tishtakalita/Multimodal-Dataset-Image_Text_Table_TimeSeries",
    "Voxel51/kitscenes-multimodal": "Voxel51/kitscenes-multimodal",
    "Multimodal-Fatima/StanfordCars_test": "Multimodal-Fatima/StanfordCars_test",
    "Multimodal-Fatima/StanfordCars_train": "Multimodal-Fatima/StanfordCars_train",
    "DAMO-NLP-SG/multimodal_textbook": "DAMO-NLP-SG/multimodal_textbook",
}

VJEPA_DATASETS: dict[str, str] = {
    "kushagrabaingaha/bmd-vjepa2-features": "kushagrabaingaha/bmd-vjepa2-features",
    "qinglinhou/sokoban-10k-vjepa2-tokenized": "qinglinhou/sokoban-10k-vjepa2-tokenized",
    "archi0918/friends-vjepa-embeddings": "archi0918/friends-vjepa-embeddings",
    "cbctr/cs2-10k-vjepa2-latents-300": "cbctr/cs2-10k-vjepa2-latents-300",
    "csusupergear/dmd2_vs_vjepa_3600": "csusupergear/dmd2_vs_vjepa_3600",
    "rookierufus/ego10k-vjepa-latents": "rookierufus/ego10k-vjepa-latents",
    "L7-Robotics/eval_smolvla_so101_conveyor_vjepa_encoder": "L7-Robotics/eval_smolvla_so101_conveyor_vjepa_encoder",
    "ThomasTheMaker/vjepa2-reacher-world-model": "ThomasTheMaker/vjepa2-reacher-world-model",
    "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo": "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo",
    "zoeloopy/lewm-vjepa21-state-moe-shared-residual-wo-pos": "zoeloopy/lewm-vjepa21-state-moe-shared-residual-wo-pos",
    "rookierufus/Video_VJEPA_CMPR_ADJ": "rookierufus/Video_VJEPA_CMPR_ADJ",
    "rookierufus/VJEPA-LATENTS-L2NORM": "rookierufus/VJEPA-LATENTS-L2NORM",
    "Adjimavo/libero_world_latents_vjepa_m4": "Adjimavo/libero_world_latents_vjepa_m4",
    "Yuchn/event-graph-vjepa-vitl-dataset": "Yuchn/event-graph-vjepa-vitl-dataset",
    "rookierufus/epic-kitchens-vjepa": "rookierufus/epic-kitchens-vjepa",
    "phi-9/epic-kitchens-vjepa": "phi-9/epic-kitchens-vjepa",
    "L7-Robotics/eval_smolvla_so101_conveyor_vjepa_predictor": "L7-Robotics/eval_smolvla_so101_conveyor_vjepa_predictor",
    "qinglinhou/sokoban-10k-vjepa2-tokenized-shards": "qinglinhou/sokoban-10k-vjepa2-tokenized-shards",
    "rookierufus/sokoban-10k-vjepa-latents": "rookierufus/sokoban-10k-vjepa-latents",
    "ThomasTheMaker/vjepa2-robot-multitask": "ThomasTheMaker/vjepa2-robot-multitask",
    "maux/community-vj21": "maux/community-vj21",
    "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings": "quastAI/behavior-1k-2025-challenge-vjepa2-vitg-demo-embeddings",
    "rookierufus/Vjepa_mamba_dataset": "rookierufus/Vjepa_mamba_dataset",
    "huyle1611/bridgev2_vjepa21_latent_sharded": "huyle1611/bridgev2_vjepa21_latent_sharded",
    "Nuntea/vjepa2-temporal-order-blindspots": "Nuntea/vjepa2-temporal-order-blindspots",
    # World-model mix (MCF v2 HLS extended)
    "facebook/jepa-wms": "facebook/jepa-wms",
    "jialei02/libero_merged_no_noops_20hz": "jialei02/libero_merged_no_noops_20hz",
    "Silicon23/mujoco-kinematics-probing": "Silicon23/mujoco-kinematics-probing",
    "Swastikr/PhysSim-VLM-Dataset": "Swastikr/PhysSim-VLM-Dataset",
    "Swastikr/PhysSim-VLM-SFT-R2-Data": "Swastikr/PhysSim-VLM-SFT-R2-Data",
    "agibot-world/AgiBotWorld2026": "agibot-world/AgiBotWorld2026",
    "HuberyLL/nms_hitl_world_model": "HuberyLL/nms_hitl_world_model",
}

DEFAULT_MULTIMODAL_DATASET = "multimodal-reasoning-lab/Zebra-CoT"
DEFAULT_VJEPA_DATASET = "rookierufus/epic-kitchens-vjepa"

TOKENIZER_OPTIONS: dict[str, str] = {
    "gpt2 (tiktoken)": "gpt2",
    "cl100k_base (tiktoken)": "cl100k_base",
    "p50k_base (tiktoken)": "p50k_base",
    "GPT-2 (HF)": "gpt2",
    "distilgpt2 (HF)": "distilgpt2",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "StorySupra-10M (8K vocab)": "SupraLabs/StorySupra-10M",
}

DEFAULT_TOKENIZER_ID = "SupraLabs/StorySupra-10M"
