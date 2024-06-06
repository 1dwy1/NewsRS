import os
import shutil
import logging

from sklearn.model_selection import train_test_split


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 分割数据集 分成训练集合7,验证1,测试2
def split_dataset():
    global train_dir
    original_dataset = '../dataset/CN_Corpus/SogouC.reduced/Reduced'
    base_dir ='../split_data/'
    if os.path.exists(base_dir):
        logging.info('`data` directory already exists.')
        return
    try:
        os.mkdir(base_dir)
        train_dir = os.path.join(base_dir, 'train')
        validate_dir=os.path.join(base_dir,'validate')
        test_dir = os.path.join(base_dir, 'test')
        category_labels = {
            'C000008': 'Finance',
            'C000010': 'IT',
            'C000013': 'Health',
            'C000014': 'Sports',
            'C000016': 'Travel',
            'C000020': 'Education',
            'C000022': 'Recruit',
            'C000023': 'Culture',
            'C000024': 'Military'
        }

        # split corpus，遍历原始数据集中的所有类别
        for cate in os.listdir(original_dataset):
            cate_dir = os.path.join(original_dataset, cate)
            file_list = os.listdir(cate_dir)
            print("cate: {}, len: {}".format(cate, len(file_list)))

            # split the files into train, test, and val sets
            train_files, test_files = train_test_split(file_list, test_size=0.3, random_state=42)
            val_files, test_files = train_test_split(test_files, test_size=2 / 3, random_state=42)

            # copy train files
            dst_dir = os.path.join(train_dir, category_labels[cate])
            os.mkdir(dst_dir)
            logging.info(f'Created main directory: {base_dir}')

            print("dst_dir (train): {}, len: {}".format(dst_dir, len(train_files)))
            for fname in train_files:
                src = os.path.join(cate_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copyfile(src, dst)
            logging.info(f'Created train directory: {train_dir}')

            # copy test files
            dst_dir = os.path.join(test_dir, category_labels[cate])
            os.mkdir(dst_dir)
            print("dst_dir (test): {}, len: {}".format(dst_dir, len(test_files)))
            for fname in test_files:
                src = os.path.join(cate_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copyfile(src, dst)
            logging.info(f'Created test directory: {test_dir}')

            # copy val files
            dst_dir = os.path.join(validate_dir, category_labels[cate])
            os.mkdir(dst_dir)
            print("dst_dir (val): {}, len: {}".format(dst_dir, len(val_files)))
            for fname in val_files:
                src = os.path.join(cate_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copyfile(src, dst)
            logging.info(f'Created validate directory: {val_files}')
        print('Corpus split DONE.')
    except Exception as e:
        logging.error(f'An error occurred while creating directories: {e}')
        # 清理已创建的目录
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)


if __name__ == '__main__':
    split_dataset()