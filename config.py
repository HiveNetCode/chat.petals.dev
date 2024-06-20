from dataclasses import dataclass
import os
from typing import Optional

import torch

from petals.constants import PUBLIC_INITIAL_PEERS


@dataclass
class ModelInfo:
    repo: str
    adapter: Optional[str] = None
    name: Optional[str] = None


MODELS = [
    #ModelInfo(repo="petals-team/StableBeluga2", name="stabilityai/StableBeluga2"),
    ModelInfo(repo="meta-llama/Llama-2-70b-chat-hf"),
    #ModelInfo(repo="huggyllama/llama-65b"),
    #ModelInfo(repo="huggyllama/llama-65b", adapter="timdettmers/guanaco-65b"),
    #ModelInfo(repo="mistralai/Mistral-7B-v0.1"),
    #ModelInfo(repo="huggyllama/llama-7b"),
    #ModelInfo(repo="tiiuae/falcon-180B-chat"),
    #ModelInfo(repo="bigscience/bloomz"),
]
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
HF_ACCESS_TOKEN = "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"
#DEFAULT_MODEL_NAME = "tiiuae/falcon-180B-chat"

#INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = []
BOOTSTRAP_PEERS = os.environ['INITIAL_PEERS']
if BOOTSTRAP_PEERS != "":
    INITIAL_PEERS.append(BOOTSTRAP_PEERS)
#else:
#    INITIAL_PEERS = ['/ip4/51.79.102.103/tcp/31337/p2p/QmT3TtHZyKGHuXzgWaC5AXscQsFRrH9jJGU8PC4YJUwD5g']

DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature
    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
