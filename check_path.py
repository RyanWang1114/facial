import os
import sys

print("=" * 70)
print("🔍 FER2013 数据集路径诊断工具")
print("=" * 70)

# 1. 获取当前脚本所在的目录和运行目录
script_dir = os.path.dirname(os.path.abspath(__file__))
current_work_dir = os.getcwd()

print(f"\n📂 脚本文件 (__file__) 所在目录: {script_dir}")
print(f"📂 终端当前运行目录 (os.getcwd()): {current_work_dir}")

# 如果两个路径不一样，解释一下
if script_dir != current_work_dir:
    print("\n⚠️  注意：脚本目录和运行目录不一样！")
    print(f"   你的 Python 脚本在: {script_dir}")
    print(f"   但你运行命令时，终端在: {current_work_dir}")
    print("   PyTorch 默认会去【运行目录】找相对路径的文件。")
    print("   我们将同时检查这两个目录。")

# 2. 列出当前运行目录下的所有文件/文件夹（帮助用户核实）
print("\n" + "-" * 70)
print(f"📋 当前运行目录 ({current_work_dir}) 下的所有内容：")
items = os.listdir(current_work_dir)
for item in sorted(items):
    item_path = os.path.join(current_work_dir, item)
    if os.path.isdir(item_path):
        print(f"  📁 [文件夹] {item}")
    else:
        print(f"  📄 [文件]   {item}")

# 3. 定义可能的数据文件夹名称（根据你的实际情况可以增加）
possible_names = [
    'archive(2)',      # 你之前说的名字
    'archive',         # 常见名字
    'fer2013',         # 常见名字
    'data',            # 常见名字
    'FER2013',         # 大写版本
    'archive (2)',     # 带空格版本（很多下载会自动加空格）
    'archive_2',       # 下划线版本
    'train'            # 万一直接把train放在了根目录下
]

# 4. 分别在 脚本目录 和 运行目录 下查找
found_paths = {}

def search_in_directory(base_dir, label):
    print("\n" + "-" * 70)
    print(f"🔎 正在搜索: {label} ({base_dir})")

    if not os.path.exists(base_dir):
        print(f"   ❌ 路径不存在: {base_dir}")
        return

    for name in possible_names:
        test_path = os.path.join(base_dir, name)
        if os.path.exists(test_path):
            if os.path.isdir(test_path):
                print(f"   ✅ 找到可能的文件夹: {test_path}")
                found_paths[test_path] = name
            else:
                print(f"   ⚠️  找到同名但非文件夹: {test_path}")

# 搜索脚本目录
search_in_directory(script_dir, "脚本所在目录")

# 搜索运行目录（如果和脚本目录不同）
if script_dir != current_work_dir:
    search_in_directory(current_work_dir, "终端运行目录")

# 5. 如果找到了，深度检查内部结构
if found_paths:
    print("\n" + "-" * 70)
    print("📦 找到的可能路径，开始深度检查：")

    for path, name in found_paths.items():
        print(f"\n   🗂️  检查: {path}")

        # 检查子目录
        sub_dirs = ['train', 'test', 'Training', 'Testing', 'Train', 'Test']
        for sub in sub_dirs:
            sub_path = os.path.join(path, sub)
            if os.path.exists(sub_path) and os.path.isdir(sub_path):
                print(f"      ✅ 找到子目录: {sub}")
                # 进一步看里面的表情文件夹
                try:
                    emotion_folders = [f for f in os.listdir(sub_path)
                                      if os.path.isdir(os.path.join(sub_path, f))]
                    if emotion_folders:
                        print(f"         📊 包含 {len(emotion_folders)} 个情绪类别: {', '.join(emotion_folders[:5])}{'...' if len(emotion_folders)>5 else ''}")
                        # 统计图片数量
                        total_images = 0
                        for ef in emotion_folders:
                            ef_path = os.path.join(sub_path, ef)
                            img_count = len([img for img in os.listdir(ef_path)
                                           if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
                            total_images += img_count
                            print(f"            - {ef}: {img_count} 张图片")
                        print(f"         📸 该子目录图片总数: {total_images}")
                    else:
                        print(f"         ⚠️  {sub} 目录为空或没有情绪子文件夹")
                except Exception as e:
                    print(f"         ❌ 读取失败: {e}")

else:
    print("\n" + "-" * 70)
    print("❌ 在所有路径下都没有找到数据文件夹！")
    print("\n💡 建议操作：")
    print("   1. 检查你解压的文件夹名字是否和上面的 'possible_names' 列表匹配。")
    print("   2. 如果名字不一样，把正确的名字加到 possible_names 列表里。")
    print("   3. 确保文件夹里有 'train' 和 'test' 子文件夹。")

# 6. 给出明确的建议
print("\n" + "-" * 70)
print("🏁 诊断结束。")
print("\n如果找到了正确的路径，你应该把上面 '✅ 找到' 对应的完整路径，")
print("替换到 model_lstm.py 的 data_root 变量里。")

print("\n✅ 你可以把上面的输出结果完整地复制粘贴给我，我帮你分析下一步。")
