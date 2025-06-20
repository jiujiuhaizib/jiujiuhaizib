# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
model = AutoModelForSequenceClassification.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "正面" if predicted_class == 1 else "负面"


# 选择对应句子
movie = "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。"
food = "食物分量十足，性价比超高，吃得很满足。"

# 执行预测
movie_sentiment = predict_sentiment(movie)
food_sentiment = predict_sentiment(food)

# 输出
print(f"影评句子：{movie}  情感倾向：{movie_sentiment}")
print(f"外卖评价：{food}  情感倾向：{food_sentiment}")