import joblib
import json
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'\[rm\]', '', text)
    text = re.sub(r'[^a-z0-9\s\.\,]', '', text)
    return text

def extract_storage_regex(text):
    """重現 Notebook 中的 extract_storage 邏輯"""
    text = str(text).lower()
    match_tb = re.search(r'(\d+)\s*tb', text)
    if match_tb: return int(match_tb.group(1)) * 1024
    matches_gb = re.findall(r'(\d+)\s*gb', text)
    if matches_gb: return max([int(x) for x in matches_gb])
    return 0

class AdvancedFeatureExtractor:
    @staticmethod
    def extract_ram(text):
        match = re.search(r'\b(4|8|12|16|24|32|64)\s*[gG][bB]?\s*(ram|memory)\b', str(text))
        return int(match.group(1)) if match else 0

    @staticmethod
    def extract_cpu_score(text):
        text = str(text).lower()
        if any(x in text for x in ['m1', 'm2', 'm3', 'i9', 'ryzen 9', 'xeon', 'threadripper']): return 5
        if any(x in text for x in ['i7', 'ryzen 7']): return 4
        if any(x in text for x in ['i5', 'ryzen 5']): return 3
        if any(x in text for x in ['i3', 'ryzen 3']): return 2
        return 1

    @staticmethod
    def is_pro_model(text):
        text = str(text).lower()
        keywords = ['pro', 'max', 'ultra', 'gaming', 'edition', 'surface book', 'macbook']
        return 1 if any(k in text for k in keywords) else 0

    @staticmethod
    def extract_resolution(text):
        text = str(text).lower()
        if any(x in text for x in ['4k', 'uhd', 'retina', '2160p']): return 2
        if any(x in text for x in ['1080p', 'fhd', 'full hd']): return 1
        return 0

    @staticmethod
    def extract_product_series(text):
        text = str(text).lower()
        if any(x in text for x in ['macbook pro', 'alienware', 'rog', 'razer blade', 'surface book', 'precision', 'zbook', 'msi gt', 'aorus']): return 3
        elif any(x in text for x in ['macbook', 'surface pro', 'xps', 'spectre', 'zenbook', 'elitebook', 'thinkpad x1', 'latitude', 'portege', 'yoga']): return 2
        elif any(x in text for x in ['inspiron', 'pavilion', 'aspire', 'ideapad', 'surface go', 'chromebook']): return 1
        return 0

    @staticmethod
    def extract_gpu_score(text):
        text = str(text).lower()
        if any(x in text for x in ['rtx 4', 'rtx 3', 'rtx 2', 'gtx 1080', 'gtx 1070', 'rtx 20', 'titan', 'quadro rtx']): return 5
        if any(x in text for x in ['gtx 1060', 'gtx 1660', 'rtx 3050', 'gtx 980', 'quadro m']): return 4
        if any(x in text for x in ['gtx 1050', 'gtx 960', 'mx350', 'radeon pro']): return 3
        if any(x in text for x in ['iris', 'vega', 'intel hd 6']): return 2
        return 1

    @staticmethod
    def extract_cpu_gen(text):
        text = str(text).lower()
        if 'm3' in text: return 14
        if 'm2' in text: return 13
        if 'm1' in text: return 12
        match = re.search(r'(core i[3579]|ryzen [3579])[\s-]?(\d{1,2})\d{2,3}', text)
        if match:
            gen = int(match.group(2))
            if 1 <= gen <= 14: return gen
        match_text = re.search(r'(\d{1,2})th gen', text)
        return int(match_text.group(1)) if match_text else 0

class ModernBertService:
    def __init__(self, model_name='answerdotai/ModernBERT-base'):
        self.device = torch.device("cpu") 
        print(f"Loading BERT: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def extract(self, text):
        inputs = self.tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu().numpy()

class PricePredictor:
    def __init__(self, model_dir="models/"):
        print("Loading XGBoost models & Stats...")
        self.clf = joblib.load(f'{model_dir}xgb_classifier.joblib')
        self.reg_low = joblib.load(f'{model_dir}xgb_reg_low.joblib')
        self.reg_high = joblib.load(f'{model_dir}xgb_reg_high.joblib')
        
        with open(f'{model_dir}imputation_stats.json', 'r') as f:
            stats = json.load(f)
            self.storage_medians = {int(k): v for k, v in stats['storage_medians'].items()}
            self.ram_medians = {int(k): v for k, v in stats['ram_medians'].items()}
            
        self.bert = ModernBertService()

    def predict(self, text, condition_id=3):
        processed_text = clean_text(text)
        bert_emb = self.bert.extract(processed_text)

        feat_storage = AdvancedFeatureExtractor.extract_storage_advanced(processed_text) if hasattr(AdvancedFeatureExtractor, 'extract_storage_advanced') else extract_storage_regex(processed_text)
        feat_ram = AdvancedFeatureExtractor.extract_ram(processed_text)
        feat_cpu = AdvancedFeatureExtractor.extract_cpu_score(processed_text)
        feat_pro = AdvancedFeatureExtractor.is_pro_model(processed_text)
        feat_res = AdvancedFeatureExtractor.extract_resolution(processed_text)
        feat_series = AdvancedFeatureExtractor.extract_product_series(processed_text)
        feat_gpu = AdvancedFeatureExtractor.extract_gpu_score(processed_text)
        feat_gen = AdvancedFeatureExtractor.extract_cpu_gen(processed_text)

        if feat_storage == 0: feat_storage = self.storage_medians.get(feat_cpu, 128)
        if feat_ram == 0: feat_ram = self.ram_medians.get(feat_cpu, 4)

        numerical_feats = np.array([[
            feat_storage, condition_id, feat_ram, feat_cpu, 
            feat_pro, feat_res, feat_series, feat_gpu, feat_gen
        ]])
        
        X_input = np.hstack([bert_emb, numerical_feats])

        prob_high = self.clf.predict_proba(X_input)[:, 1][0]
        prob_high_boosted = np.sqrt(prob_high)
        
        pred_low = np.expm1(self.reg_low.predict(X_input)[0])
        pred_high = np.expm1(self.reg_high.predict(X_input)[0])
        
        # 混合
        final_price = (prob_high_boosted * pred_high) + ((1 - prob_high_boosted) * pred_low)
        
        # C. 商業規則修正
        text_lower = processed_text.lower()
        if 'surface pro' in text_lower and final_price < 300: final_price = 300.0
        if any(c in text_lower for c in ['m1', 'm2', 'm3']) and final_price < 800: final_price = 800.0

        return round(float(final_price), 2), processed_text