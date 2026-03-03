import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path

class ESMFeatureExtractor:
    """
    使用本地 ESM-2 模型，为蛋白质序列离线提取 CLS 与 tokens 特征。
    - 以 CSV 中的“原始序列字符串”作为字典 key。
    - 批内按最长样本 padding，超长 truncation 到 1024（含 CLS/EOS）。
    """
    def __init__(self):
        self.RAW_DATA_FILE = "/home/u2308283088/libiao/2026_1_13/drugbank.csv"
        self.ESM2_EMBEDDINGS_FILE = "/home/u2308283088/libiao/2026_1_13/drugbank_esm2_embeddings.pt"
        self.MAX_PROTEIN_LEN = 1024
        self.BATCH_SIZE = 32
        self.ESM2_MODEL_PATH = "/home/u2308283088/libiao/esm2_model/esm2_150m"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        model_path = Path(self.ESM2_MODEL_PATH)
        print(f"模型路径: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        print("模型加载成功！")

    def extract_and_save_features(self):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        df = pd.read_csv(self.RAW_DATA_FILE)
        sequences = df["sequence"].dropna().unique().tolist()
        sequences = [s for s in sequences if s]
        print(f"蛋白质序列总数: {len(sequences)}")

        protein_embeddings = {}
        failed_samples = []

        Path(self.ESM2_EMBEDDINGS_FILE).parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            pbar = tqdm(range(0, len(sequences), self.BATCH_SIZE), desc="Extracting ESM-2")
            for i in pbar:
                batch = sequences[i:i+self.BATCH_SIZE]
                try:
                    enc = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_PROTEIN_LEN
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out = self.model(**enc)
                    H = out.last_hidden_state.detach().cpu()
                    lens = enc["attention_mask"].sum(1)
                    D = H.shape[-1]

                    for j, seq_string in enumerate(batch):
                        L = int(lens[j].item())
                        cls_vec = H[j, 0, :].clone()
                        if L > 2:
                            tokens = H[j, 1:L-1, :].clone()
                        else:
                            tokens = torch.empty((0, D), dtype=H.dtype)
                        protein_embeddings[seq_string] = {
                            "global": cls_vec,
                            "tokens": tokens
                        }
                except Exception as e:
                    failed_samples.extend(batch)
                    print(f"\n[BatchError] {batch[0][:30]}... -> {e}")

        torch.save(protein_embeddings, self.ESM2_EMBEDDINGS_FILE)
        print(f"特征提取完成，成功: {len(protein_embeddings)}，失败: {len(failed_samples)}")
        print(f"特征已保存到: {self.ESM2_EMBEDDINGS_FILE}")
        if failed_samples:
            print(f"失败样本数: {len(failed_samples)}，示例: {failed_samples[:3]}")
        # 打印一个样本的CLS和tokens特征维度
        if len(protein_embeddings) > 0:
            first_seq = next(iter(protein_embeddings))
            cls_shape = protein_embeddings[first_seq]["global"].shape
            tokens_shape = protein_embeddings[first_seq]["tokens"].shape
            print(f"示例序列: {first_seq[:30]}...")
            print(f"CLS特征维度: {cls_shape}")
            print(f"Tokens特征维度: {tokens_shape}")

def main():
    ESMFeatureExtractor().extract_and_save_features()

if __name__ == "__main__":
    main()