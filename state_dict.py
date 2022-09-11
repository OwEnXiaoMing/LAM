import torch

dict1 = torch.load('D:\pycharmworkspace\LAM\ModelZoo\ModelZoo\models\EDSR-64-16_15000.pth')
#print(dict1.keys())
dict2 = torch.load('D:\pycharmworkspace\LAM\ModelZoo\ModelZoo\models\edsr_s.pt')
#print(dict2.keys())
dict3 = torch.load('D:\pycharmworkspace\LAM\ModelZoo\ModelZoo\models\edsr_baseline_x4.pt')
torch.save()
#print(dict3.keys())
#print([dict1[a].size() for a in dict1.keys()]==[dict2[a].size() for a in dict2.keys()])
