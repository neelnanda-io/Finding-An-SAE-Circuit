# %%
import os

os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
python_prompt = """15. Georgia
16. Mississippi
17. Daniel
"""
python_answer = "18"
utils.test_prompt(python_prompt, python_answer, model, prepend_space_to_answer=False)
# %%
# Layer 19, feature 18
from transformer_lens import HookedSAETransformer
from transformer_lens import HookedSAE, HookedSAEConfig
from transformer_lens.utils import download_file_from_hf

model = HookedSAETransformer.from_pretrained("gpt2-small")

def attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg):
    new_cfg = {
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "hook_name": attn_sae_cfg["act_name"],
    }
    return HookedSAEConfig.from_dict(new_cfg)
auto_encoder_runs = [
    "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
]

hf_repo = "ckkissane/attn-saes-gpt2-small-all-layers"

att_sae_cfg = download_file_from_hf(hf_repo, f"{auto_encoder_runs[0]}_cfg.json")
cfg = attn_sae_cfg_to_hooked_sae_cfg(att_sae_cfg)

state_dict = download_file_from_hf(hf_repo, f"{auto_encoder_runs[0]}.pt", force_is_torch=True)

# from typing import Mapping, Callable
# assert isinstance(state_dict, Mapping[str, Any])
# assert isinstance(model.attach_sae, Callable)
hooked_sae = HookedSAE(cfg)
hooked_sae.load_state_dict(state_dict) # type: ignore
model.add_sae(hooked_sae) # type: ignore
# %%
clean_prompt = "14. Washington 15. California 16. Oregon"
clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corr_prompt = "14. Washington 15. California 16. Daniel"
corr_logits, corr_cache = model.run_with_cache(corr_prompt)

SEVENTEEN = model.to_single_token(" 17")
print(clean_logits.softmax(-1)[0, -1, SEVENTEEN])
print(corr_logits.softmax(-1)[0, -1, SEVENTEEN])

str_tokens = nutils.process_tokens_index(clean_prompt)

line(
    [
        clean_cache["blocks.9.attn.hook_z.hook_sae_acts_pre"][0, :, 18],
        corr_cache["blocks.9.attn.hook_z.hook_sae_acts_pre"][0, :, 18],
    ],
    line_labels=["clean", "corr"],
    x=str_tokens
)
# %%
# Build a better dataset
us_states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]


# %%
# Look at the attention patterns of the L9 heads

# Do direct feature attribution, recursing further and further back

# Patch the attention patterns and see how much difference in behaviour this removes
