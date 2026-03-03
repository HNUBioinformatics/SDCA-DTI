import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path

class ChemBERTFeatureExtractor:
    """
    使用本地 ChemBERTa 模型为 SMILES 提取:
      global: 第0位置 CLS/<s> 向量
      tokens: 去除首尾特殊token后的有效序列向量
    """
    def __init__(self):
        self.RAW_DATA_FILE = "/home/u2308283088/libiao/2026_1_13/drugbank.csv"
        self.CHEMBERT_EMBEDDINGS_FILE = "/home/u2308283088/libiao/2026_1_13/drugbank_chembert_embeddings.pt"
        self.MAX_DRUG_LEN = 256
        self.BATCH_SIZE = 32
        self.CHEMBERT_MODEL_PATH = "/home/u2308283088/libiao/chemberta_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        model_path = Path(self.CHEMBERT_MODEL_PATH)
        print(f"模型路径: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            self.MAX_DRUG_LEN = min(self.MAX_DRUG_LEN, max_pos)
        print(f"使用 max_length={self.MAX_DRUG_LEN}")

    def extract_and_save_features(self):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        df = pd.read_csv(self.RAW_DATA_FILE)
        if "smiles" not in df.columns:
            raise KeyError("CSV 未找到 'smiles' 列")
        smiles_list = df["smiles"].dropna().unique().tolist()
        smiles_list = [s for s in smiles_list if isinstance(s, str) and s.strip()]
        print(f"唯一 SMILES 数: {len(smiles_list)}")

        embeddings = {}
        failed = []
        Path(self.CHEMBERT_EMBEDDINGS_FILE).parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            pbar = tqdm(range(0, len(smiles_list), self.BATCH_SIZE), desc="Extracting ChemBERTa")
            for i in pbar:
                batch = smiles_list[i:i+self.BATCH_SIZE]
                try:
                    enc = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_DRUG_LEN
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out = self.model(**enc)
                    hidden = out.last_hidden_state.detach().cpu()     # [B, L, H]
                    attn = enc["attention_mask"].cpu()                # [B, L]
                    _, L, H = hidden.shape

                    for j, smi in enumerate(batch):
                        valid_len = int(attn[j].sum().item())          # 含特殊token
                        cls_vec = hidden[j, 0, :].clone()
                        if valid_len > 2:
                            tokens = hidden[j, 1:valid_len-1, :].clone()
                        else:
                            tokens = torch.empty((0, H), dtype=hidden.dtype)
                        embeddings[smi] = {
                            "global": cls_vec,
                            "tokens": tokens
                        }
                except Exception as e:
                    failed.extend(batch)
                    print(f"[BatchError] {batch[0][:30]}... -> {e}")

        torch.save(embeddings, self.CHEMBERT_EMBEDDINGS_FILE)
        print(f"完成: 成功 {len(embeddings)}，失败 {len(failed)}")
        print(f"保存路径: {self.CHEMBERT_EMBEDDINGS_FILE}")
        if failed:
            print(f"失败示例: {failed[:3]}")
        if embeddings:
            first = next(iter(embeddings))
            print(f"示例 SMILES: {first}")
            print(f"global shape: {tuple(embeddings[first]['global'].shape)}")
            print(f"tokens shape: {tuple(embeddings[first]['tokens'].shape)}")

def main():
    ChemBERTFeatureExtractor().extract_and_save_features()
    """
 结果字典: {smiles: {"global": Tensor[H], "tokens": Tensor[T,H]}}
    """
if __name__ == "__main__":
    main()