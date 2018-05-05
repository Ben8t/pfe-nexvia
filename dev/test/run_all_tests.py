import os

test_file_list = []
for root, dirs, files in os.walk("dev/"):
    for file in files:
        if file.endswith(".py") and file != "__init__.py" and file.startswith("test"):
            test_file_list.append(os.path.join(root, file))
for file in test_file_list:
    os.system('{} {}'.format('python', file))
