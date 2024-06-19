import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from translate import Translator
from tqdm import tqdm, trange
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
print("Загрузка данных")
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
print("Данные загружены")
print("Перевод сообщений на русский язык")
translator = Translator(to_lang="ru")
data['message_ru'] = data['message'].apply(lambda x: translator.translate(x))
print("Перевод завершен")
data['message_combined'] = data['message'] + ' ' + data['message_ru']

print("Разделение данных")
X_train, X_test, y_train, y_test = train_test_split(data['message_combined'], data['label'], test_size=0.2, random_state=42)

print("Преобразование текстовых данных в TF-IDF")
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

print("Преобразование меток в числовые значения")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("Преобразование данных в ткнзоры")
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        return out


input_dim = X_train_tfidf.shape[1]
model = SpamClassifier(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5000
print("Начало обучения")
for epoch in trange(num_epochs, desc="Epochs"):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

print("Обучение завершено")

print("Оценка модели")
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())
print(f'Accuracy: {accuracy:.4f}')

print("Классификационный отчет:")
print(classification_report(y_test_tensor.cpu(), predicted.cpu(), target_names=encoder.classes_))

def classify_message(model, message, vectorizer, translator, encoder):
    message_ru = translator.translate(message)
    message_combined = message + ' ' + message_ru
    message_tfidf = vectorizer.transform([message_combined]).toarray()
    message_tensor = torch.tensor(message_tfidf, dtype=torch.float32)
    print("Классификация сообщения...")
    with torch.no_grad():
        output = model(message_tensor)
        _, predicted = torch.max(output, 1)
    label = encoder.inverse_transform(predicted.cpu().numpy())[0]
    return label

translator = Translator(to_lang="ru")
while True:
    user_message = input("Введите сообщение для классификации (exit для выхода): ")
    if user_message.lower() == 'exit':
        break
    label = classify_message(model, user_message, vectorizer, translator, encoder)
    print(f'Сообщение классифицировано как: {label}')
