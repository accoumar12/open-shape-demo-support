import json
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


meta = json.load(
    open(hf_hub_download("OpenShape/openshape-objaverse-embeddings", "objaverse_meta.json", token=True, repo_type='dataset'))
)
# {
# "u": "94db219c315742909fee67deeeacae15",
# "name": "knife", "like": 0, "view": 35,
# "tags": ["game-ready", "damascus", "damascus_steel", "kabar-knife", "knife", "blender", "blender3d", "gameready"],
# "cats": ["weapons-military"],
# "img": "https://media.sketchfab.com/models/94db219c315742909fee67deeeacae15/thumbnails/c0bbbd475d264ff2a92972f5115564ee/0cd28a130ebd4d9c9ef73190f24d9a42.jpeg",
# "desc": "", "faces": 1724, "size": 11955, "lic": "by",
# "glb": "glbs/000-000/94db219c315742909fee67deeeacae15.glb"
# }
meta = {x['u']: x for x in meta['entries']}
deser = torch.load(
    hf_hub_download("OpenShape/openshape-objaverse-embeddings", "objaverse.pt", token=True, repo_type='dataset'), map_location='cpu'
)
us = deser['us']
feats = deser['feats']


def retrieve(embedding, top):
    sims = []
    embedding = F.normalize(embedding.detach().cpu(), dim=-1).squeeze()
    for chunk in torch.split(feats, 10240):
        sims.append(embedding @ F.normalize(chunk.float(), dim=-1).T)
    sims = torch.cat(sims)
    sims, idx = torch.topk(sims, top * 2)
    results = []
    for i, sim in zip(idx, sims):
        if us[i] in meta:
            results.append(dict(meta[us[i]], sim=sim))
        if len(results) >= top:
            break
    return results
