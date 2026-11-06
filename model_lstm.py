import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

class FER2013_VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=10, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []

        for emotion_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, emotion_folder)
            if not os.path.isdir(folder_path):
                continue
            images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                           if f.endswith(('.jpg', '.png', '.jpeg'))])

            for i in range(0, len(images) - seq_len + 1, seq_len):
                self.samples.append((images[i:i+seq_len], emotion_folder))

        self.label_map = {'angry':0, 'disgust':1, 'fear':2, 'happy':3,
                         'sad':4, 'surprise':5, 'neutral':6}
        print(f"✅ 从 {root_dir} 生成了 {len(self.samples)} 个视频片段（每段 {seq_len} 帧）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label_str = self.samples[idx]
        label = self.label_map[label_str]

        frames = []
        for path in img_paths:
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        video = torch.stack(frames, dim=0)
        return video, label


class EmotionCNN_LSTM(nn.Module):
    def __init__(self, seq_len=10):
        super(EmotionCNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_cnn = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(input_size=256 * 6 * 6,
                           hidden_size=256,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.3)

        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size * seq_len, C, H, W)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_cnn(x)

        x = x.view(batch_size * seq_len, -1)

        x = x.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :]

        x = self.fc(x)
        return x


print("="*60)
print("🚀 启动 FER2013 视频序列训练 (CNN + LSTM)")
print("="*60)

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📂 脚本所在目录: {script_dir}")

possible_names = [
    'archive (2)',
    'archive(2)',
    'archive',
    'fer2013',
    'data',
    'FER2013'
]
data_root = os.environ.get('FER2013_DATA_DIR')

if data_root is not None and not os.path.isdir(data_root):
    raise FileNotFoundError(f"FER2013_DATA_DIR does not exist: {data_root}")

for name in possible_names:
    if data_root is not None:
        break
    test_path = os.path.join(script_dir, name)
    if os.path.exists(test_path) and os.path.isdir(test_path):
        data_root = test_path
        print(f"✅ 找到数据文件夹: {name}")
        break

if data_root is None:
    print("\n❌ 错误：找不到数据文件夹！")
    print(f"当前目录（{script_dir}）下的所有文件夹：")
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}")
    raise FileNotFoundError("找不到数据文件夹，请检查路径。")

def find_subdir(base_dir, candidate_names):
    for name in candidate_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path) and os.path.isdir(path):
            return path
    try:
        all_items = os.listdir(base_dir)
        for item in all_items:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                if any(name.lower() == item.lower() for name in candidate_names):
                    return item_path
    except:
        pass
    return None

train_dir = find_subdir(data_root, ['train', 'Train', 'training', 'Training'])
test_dir = find_subdir(data_root, ['test', 'Test', 'testing', 'Testing'])

if train_dir is None:
    raise FileNotFoundError(f"❌ 在 {data_root} 下找不到训练子目录 (train/Train)")
if test_dir is None:
    raise FileNotFoundError(f"❌ 在 {data_root} 下找不到测试子目录 (test/Test)")

print(f"📁 训练数据路径: {train_dir}")
print(f"📁 测试数据路径: {test_dir}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = FER2013_VideoDataset(train_dir, seq_len=10, transform=transform)
test_dataset = FER2013_VideoDataset(test_dir, seq_len=10, transform=transform)

BATCH_SIZE = 32
SEQ_LEN = 10

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"📦 训练集批次数: {len(train_loader)}，测试集批次数: {len(test_loader)}")
print(f"⚙️  Batch Size: {BATCH_SIZE}，序列长度: {SEQ_LEN}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN_LSTM(seq_len=SEQ_LEN).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"🖥️  使用设备: {device}")
print(f"📊 模型参数量: {total_params / 1e6:.2f}M")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


EPOCHS = 20
print(f"\n🏋️ 开始训练，总 Epochs: {EPOCHS}")
print("-"*60)

train_losses = []
test_accuracies = []
learning_rates = []

total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            avg_loss_so_far = running_loss / (batch_idx + 1)
            print(f'  [Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx+1}/{num_batches} | '
                  f'Loss: {loss.item():.4f} | Avg Loss: {avg_loss_so_far:.4f} | LR: {current_lr:.6f}')

    avg_train_loss = running_loss / num_batches
    train_losses.append(avg_train_loss)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    scheduler.step(avg_test_loss)
    new_lr = optimizer.param_groups[0]['lr']

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_min = int(epoch_duration // 60)
    epoch_sec = int(epoch_duration % 60)

    gpu_mem = ""
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        gpu_mem = f" | GPU: {mem_allocated:.0f}MB/{mem_reserved:.0f}MB"

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] 完成 | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Test Loss: {avg_test_loss:.4f} | "
          f"Test Acc: {accuracy:.2f}% | "
          f"LR: {new_lr:.6f} | "
          f"⏱️ 耗时: {epoch_min}分{epoch_sec}秒{gpu_mem}")
    print("-"*60)

total_end_time = time.time()
total_duration = total_end_time - total_start_time
total_min = int(total_duration // 60)
total_sec = int(total_duration % 60)

print("\n" + "="*60)
print(f"🎉 全部训练完成！总耗时: {total_min} 分 {total_sec} 秒")
print(f"📈 最终测试准确率: {test_accuracies[-1]:.2f}%")
print("="*60)


torch.save(model.state_dict(), os.path.join(script_dir, 'fer2013_lstm_model.pth'))
print("💾 模型已保存为 fer2013_lstm_model.pth")


plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(range(1, EPOCHS+1), train_losses, 'b-', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(range(1, EPOCHS+1), test_accuracies, 'r-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(range(1, EPOCHS+1), learning_rates, 'g-', label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.title('Learning Rate Schedule')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ 全部完成！")
