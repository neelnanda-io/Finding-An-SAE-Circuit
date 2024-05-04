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
# Layer 9, feature 18
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
states = pd.DataFrame({"state": us_states, "num_tokens": [len(model.to_str_tokens(" "+s, prepend_bos=False)) for s in us_states]})
nutils.show_df(states)
us_states = states.query("num_tokens==1").state.to_list()
print(us_states)

names = ["Daniel", "Jacob", "Bart", "John", "Matt", "Anne", "Alice", "Mary", "Tom"]

batch_size = 64
random.seed(512)
state_1 = random.choices(us_states, k=batch_size)
state_2 = random.choices(us_states, k=batch_size)
state_3 = random.choices(us_states, k=batch_size)
name = random.choices(names, k=batch_size)

TEMPLATE = "14. {} 15. {} 16. {}"
clean_prompts = []
corr_prompts = []
for s1, s2, s3, n in zip(state_1, state_2, state_3, name):
    clean_prompts.append(TEMPLATE.format(s1, s2, s3))
    corr_prompts.append(TEMPLATE.format(s1, s2, n))
clean_tokens = model.to_tokens(clean_prompts)
corr_tokens = model.to_tokens(corr_prompts)
# %%
clean_logits, clean_cache = model.run_with_cache(clean_prompts)
corr_logits, corr_cache = model.run_with_cache(corr_prompts)
line(
    [
        clean_logits.log_softmax(dim=-1)[:, -1, SEVENTEEN],
        corr_logits.log_softmax(dim=-1)[:, -1, SEVENTEEN],
    ],
    line_labels=["clean", "corr"],
    title="Log Prob of seventeen"
)
SAE_NAME = "blocks.9.attn.hook_z.hook_sae_acts_pre"
line(
    [
        clean_cache[SAE_NAME][:, -1, 18],
        corr_cache[SAE_NAME][:, -1, 18],
    ],
    line_labels=["clean", "corr"],
    title="SAE activations"
)
# %%
# Look at the attention patterns of the L9 heads
STR_LABELS = [
    "BOS", "14", ".1", "S1", " 15", ".2", "S2", " 16", ".3", "S3/Name"
]
imshow([
    clean_cache["attn", 9][:, 1, -1, :],
    corr_cache["attn", 9][:, 1, -1, :],
], facet_col=0, facet_labels=["clean", "corr"], x=STR_LABELS, xaxis="Prompt", yaxis="Batch", title="L9H1 (Successor) Attention from final token pos")
# %%
# Do direct feature attribution, recursing further and further back
f_id = 18
w_enc = hooked_sae.W_enc[:, f_id]
w_dec = hooked_sae.W_dec[f_id]
b_enc = hooked_sae.b_enc
b_dec = hooked_sae.b_dec

clean_sae_acts = clean_cache[SAE_NAME][:, -1, f_id]
HEAD = 1
LAYER = 9
POS = -1
clean_z = clean_cache["blocks.9.attn.hook_z.hook_sae_input"][:, POS, :, :]
clean_z = einops.rearrange(clean_z, "batch head d_head -> batch (head d_head)")

sae_act_v1 = (clean_z - b_dec) @ w_enc + b_enc[f_id]
sae_act_v2 = (clean_z.squeeze()) @ w_enc + b_enc[f_id]
line([clean_sae_acts, sae_act_v1, sae_act_v2], line_labels=["clean", "w/ -b_dec", "w/o -b_dec"], title="SAE Acts")
# %%
n_heads = 12
line(w_enc.reshape(n_heads, d_head).norm(dim=-1))
# %%
bias = b_enc[f_id] - b_dec @ w_enc
scatter(clean_sae_acts, clean_z @ w_enc + bias, include_diag=True)
# %%
# DFA by head
corr_z = corr_cache["blocks.9.attn.hook_z.hook_sae_input"][:, POS, :, :]
clean_z = clean_cache["blocks.9.attn.hook_z.hook_sae_input"][:, POS, :, :]
w_enc_head = w_enc.reshape(n_heads, d_head)
diff_z = clean_z - corr_z
clean_head_dfa = einops.einsum(clean_z, w_enc_head, "batch head d_head, head d_head -> batch head")
corr_head_dfa = einops.einsum(corr_z, w_enc_head, "batch head d_head, head d_head -> batch head")
diff_head_dfa = einops.einsum(diff_z, w_enc_head, "batch head d_head, head d_head -> batch head")
imshow([clean_head_dfa, corr_head_dfa, diff_head_dfa], facet_col=0, facet_labels=["clean", "corr", "diff"], yaxis="Batch", xaxis="Head", title="Head DFA")

# %%
# DFA by head and src
w_enc_l9h1 = w_enc_head[HEAD]

clean_l9h1_v = clean_cache["v", 9][:, :, HEAD, :]
clean_l9h1_attn = clean_cache["attn", 9][:, HEAD, POS, :]
clean_l9h1_decomp_z = clean_l9h1_v * clean_l9h1_attn[:, :, None]
clean_l9h1_decomp_z_dfa = clean_l9h1_decomp_z @ w_enc_l9h1

corr_l9h1_v = corr_cache["v", 9][:, :, HEAD, :]
corr_l9h1_attn = corr_cache["attn", 9][:, HEAD, POS, :]
corr_l9h1_decomp_z = corr_l9h1_v * corr_l9h1_attn[:, :, None]
corr_l9h1_decomp_z_dfa = corr_l9h1_decomp_z @ w_enc_l9h1

diff_l9h1_decomp_z_dfa = clean_l9h1_decomp_z_dfa - corr_l9h1_decomp_z_dfa

imshow([clean_l9h1_decomp_z_dfa, corr_l9h1_decomp_z_dfa, diff_l9h1_decomp_z_dfa], facet_col=0, facet_labels=["clean", "corr", "diff"], title="L9H1 DFA by src", x=str_tokens, yaxis="Batch")
# %%
# Decompose DFA on src residual stream
clean_cache.cache_dict["blocks.9.attn.hook_z"] = clean_cache["blocks.9.attn.hook_z.hook_sae_output"]
corr_cache.cache_dict["blocks.9.attn.hook_z"] = corr_cache["blocks.9.attn.hook_z.hook_sae_output"]
# %%

clean_decomp_src_resid, labels = clean_cache.get_full_resid_decomposition(9, expand_neurons=False, return_labels=True, pos_slice=-3, apply_ln=True)
corr_decomp_src_resid, labels = corr_cache.get_full_resid_decomposition(9, expand_neurons=False, return_labels=True, pos_slice=-3, apply_ln=True)
# %%
W_V = model.W_V[LAYER, HEAD]
W_O = model.W_O[LAYER, HEAD]
clean_decomp_src_dfa = (clean_decomp_src_resid @ (W_V @ w_enc_l9h1)) * clean_l9h1_attn[:, -3]
corr_decomp_src_dfa = (corr_decomp_src_resid @ (W_V @ w_enc_l9h1)) * corr_l9h1_attn[:, -3]
diff_decomp_src_dfa = clean_decomp_src_dfa - corr_decomp_src_dfa

line([
    clean_decomp_src_dfa.mean(-1), 
    corr_decomp_src_dfa.mean(-1), 
    diff_decomp_src_dfa.mean(-1), 
    ],
    line_labels=["clean", "corr", "diff"], title="Decomposed DFA on src residual stream of 16 via L9H1", x=labels, yaxis="DFA")

# %%
# How consistent are the keys of L9H1
for i in range(2, 6):
    keys = clean_cache["k", 9][:, -3, i, :]
    imshow(nutils.cos(keys[:, None], keys[None, :]), title=f"L9H{i}")
# %%
keys = clean_cache["k", 9][:, -3, 1, :]
ave_key = keys.mean(0)
look_at_16_dir = ave_key @ model.W_Q[LAYER, HEAD].T

clean_decomp_dest_resid, labels = clean_cache.get_full_resid_decomposition(9, expand_neurons=False, return_labels=True, pos_slice=-1, apply_ln=True)
corr_decomp_dest_resid, labels = corr_cache.get_full_resid_decomposition(9, expand_neurons=False, return_labels=True, pos_slice=-1, apply_ln=True)

clean_look_at_16_attr = clean_decomp_dest_resid @ look_at_16_dir
corr_look_at_16_attr = corr_decomp_dest_resid @ look_at_16_dir

diff_look_at_16_attr = clean_look_at_16_attr - corr_look_at_16_attr

line(
    [
        clean_look_at_16_attr.mean(-1),
        corr_look_at_16_attr.mean(-1),
        diff_look_at_16_attr.mean(-1),
    ],
    line_labels=["clean", "corr", "diff"],
    title="How much each component makes L9H1 look at 16",
    x=labels,
    yaxis="DFA",
)
# %%
from sae_lens import SparseAutoencoder

layer = 9 # pick a layer you want.
resid_sae = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre"
)
for name, param in resid_sae.named_parameters():
    print(name, param.shape)
# %%
clean_resids = clean_cache["resid_pre", 9][:, -1, :]
corr_resids = corr_cache["resid_pre", 9][:, -1, :]

# %%
resid_sae.to("cuda")
look_at_16_sae = resid_sae.W_dec @ look_at_16_dir
line(look_at_16_sae, title="look_at_16_sae")

clean_sae_acts = clean_resids.mean(0) @ resid_sae.W_enc + (resid_sae.b_enc - resid_sae.b_dec @ resid_sae.W_enc)
corr_sae_acts = corr_resids.mean(0) @ resid_sae.W_enc + (resid_sae.b_enc - resid_sae.b_dec @ resid_sae.W_enc)
diff_sae_acts = clean_sae_acts - corr_sae_acts

line([clean_sae_acts, corr_sae_acts, diff_sae_acts], line_labels=["clean", "corr", "diff"])

df = pd.DataFrame({
    "clean": to_numpy(clean_sae_acts),
    "corr": to_numpy(corr_sae_acts),
    "diff": to_numpy(diff_sae_acts),
    "look_at_16": to_numpy(look_at_16_sae),
})
df["fires"] = (df["clean"] > 0) | (df["corr"] > 0)
df[df["fires"]].sort_values("diff", key=np.abs, ascending=False)

# %%
df["diff"] = np.maximum(df["clean"], 0) - np.maximum(df["corr"], 0)
df["diff*attr"] = df["diff"] * df["look_at_16"]
nutils.show_df(df[df["fires"]].sort_values("diff*attr", ascending=False).head(25))
nutils.show_df(df[df["fires"]].sort_values("diff*attr", ascending=False).tail(25))
# %%
