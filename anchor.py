
from turtle import color
import pandas as pd

# read_csv = pd.read_csv('data.csv')

data_origin = [0.220488,
0.222345,
0.231696,
0.242,
0.236591,
0.233964] 
# data_origin = data_origin.values
data_improve = [0.39086,
0.402151,
0.404301,
0.4253,
0.417204,
0.409677]
# data_improve = data_improve.values
print(data_origin)

data_origin2 = [0.218837,
0.221052,
0.221023,
0.2326,
0.230421,
0.228734]

data_improve2 = [0.396682,
0.399698,
0.397436,
0.4186,
0.414781,
0.407994]


Book_ndcg= [0.248176,
0.247858,
0.251466,
0.2581,
0.247894,
0.246721]
Book_hr= [0.427419,
0.429032,
0.437097,
0.4468,
0.430108,
0.426882]
Music_ndcg= [0.420036,
0.420362,
0.420357,
0.4306,
0.424672,
0.42313]
Music_hr= [0.729658,
0.731179,
0.735741,
0.743,
0.735741,
0.738783]

import matplotlib.pyplot as plt
import numpy as np
labels = ['10', '20', '30', '40', '50', '60'] #, '100']
x = np.arange(len(labels))
width = 0.35

# color1 = np.array([218, 198, 142])
# color1 = color1 / 255
# color2 = np.array([142, 162, 101])
# color2 = color2 / 255
color3 = np.array([250, 127, 111])#([1,86,153])# [129, 184,223])#([243, 118, 74])##([113, 178, 189])
color3 = color3 / 255
# color4 = np.array([83, 143, 183])
# color4 = color4 / 255
color5 = np.array([190, 184, 220]) #([250,192,15])#([254,129,125])#([235, 221, 184]) 95,198,201
color5 = color5 / 255

fig, ax = plt.subplots(2,2)
rects1 = ax[0,0].bar(x-width/2, data_origin, width, label='Book', color = color3)
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(labels)
ax[0,0].set_xlabel('(c) Sub-distribution Analysis \n on Book-Music NDCG@10') 
# ax[0,0].set_title("")
ax[0,0].legend(fontsize= '8', loc=2)
fig.tight_layout()
ax[0,0].set_yticks(np.arange(0.0, 0.25, 0.01)) 
ax[0,0].set_ylim(0.21, 0.25)

ax2 = ax[0,0].twinx()
rects2 = ax2.bar(x+width/2, data_improve, width, label='Music', color = color5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(fontsize= '8', loc=1)
fig.tight_layout()
ax2.set_yticks(np.arange(0.0, 0.43, 0.01)) 
ax2.set_ylim(0.39, 0.43)
# rects2 = ax[0,1].bar(x+width/2, data_improve, width, label='HR@10')
# ax[0,1].set_xticks(x)
# ax[0,1].set_xticklabels(labels)
# ax[0,1].set_xlabel('(b)') 
# ax[0,1].set_title("Book")
# ax[0,1].legend()
# fig.tight_layout()
# ax[0,1].set_yticks(np.arange(0.0, 0.76, 0.01)) 
# ax[0,1].set_ylim(0.40, 0.46)

rects3 = ax[0,1].bar(x-width/2, Book_ndcg, width, label='Book', color = color3)
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(labels)
ax[0,1].set_xlabel('(d) Sub-distribution Analysis \n on Book-Movie NDCG@10')
# ax[0,1].set_title("")
ax[0,1].legend(fontsize= '8', loc=2)
fig.tight_layout()
ax[0,1].set_yticks(np.arange(0.0, 0.27, 0.01)) 
ax[0,1].set_ylim(0.23, 0.27)


ax3 = ax[0,1].twinx()
rects4 = ax3.bar(x+width/2, Music_ndcg, width, label='Movie', color = color5)
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(fontsize= '8', loc=1)
fig.tight_layout()
ax3.set_yticks(np.arange(0.0, 0.44, 0.01)) 
ax3.set_ylim(0.40, 0.44)


# fig, ax = plt.subplots(2,2)
rects1 = ax[1,0].bar(x-width/2, data_origin2, width, label='Book', color = color3)
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(labels)
ax[1,0].set_xlabel('(c) Sub-distribution Analysis \n on Book-Music HR@10') 
# ax[1,0].set_title("")
ax[1,0].legend(fontsize= '8', loc=2)
fig.tight_layout()
ax[1,0].set_yticks(np.arange(0.0, 0.25, 0.01)) 
ax[1,0].set_ylim(0.21, 0.25)

ax2 = ax[1,0].twinx()
rects2 = ax2.bar(x+width/2, data_improve2, width, label='Music', color = color5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(fontsize= '8', loc=1)
fig.tight_layout()
ax2.set_yticks(np.arange(0.0, 0.43, 0.01)) 
ax2.set_ylim(0.39, 0.43)
# rects2 = ax[0,1].bar(x+width/2, data_improve, width, label='HR@10')
# ax[0,1].set_xticks(x)
# ax[0,1].set_xticklabels(labels)
# ax[0,1].set_xlabel('(b)') 
# ax[0,1].set_title("Book")
# ax[0,1].legend()
# fig.tight_layout()
# ax[0,1].set_yticks(np.arange(0.0, 0.76, 0.01)) 
# ax[0,1].set_ylim(0.40, 0.46)

rects3 = ax[1,1].bar(x-width/2, Book_hr, width, label='Book', color = color3)
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(labels)
ax[1,1].set_xlabel('(d) Sub-distribution Analysis \n on Book-Movie HR@10')
# ax[1,1].set_title("")
ax[1,1].legend(fontsize= '8', loc=2)
fig.tight_layout()
ax[1,1].set_yticks(np.arange(0.0, 0.46, 0.01)) 
ax[1,1].set_ylim(0.42, 0.46)


ax3 = ax[1,1].twinx()
rects4 = ax3.bar(x+width/2, Music_hr, width, label='Movie', color = color5)
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(fontsize= '8', loc=1)
fig.tight_layout()
ax3.set_yticks(np.arange(0.0, 0.755, 0.01)) 
ax3.set_ylim(0.715, 0.755)


plt.savefig('anchor.svg')
plt.show()


# data_improvez = [0.420968, 0.435484, 0.446774, 0.442473]
# data_improve2z = [0.725475,0.736882,0.742966,0.734221]
# Book_hr= [0.398387,0.421505,0.409677,0.412903]
# Music_hr= [0.37222,0.38914,0.409502,0.396682]
# labels = ['16', '32', '64', '128']
# x = np.arange(len(labels))
# width = 0.35

# color3 = np.array([251,132,2])
# color3 = color3 / 255
# # color4 = np.array([83, 143, 183])
# # color4 = color4 / 255
# color5 = np.array([70,172,202])#([235, 221, 184]) 95,198,201
# color5 = color5 / 255

# rects5 = ax[1,0].bar(x-width/2, data_improvez, width, label='Book', color = color3)
# ax[1,0].set_xticks(x)
# ax[1,0].set_xticklabels(labels)
# ax[1,0].set_xlabel('(c)')
# ax[1,0].set_title("Book-Movie HR@10")
# ax[1,0].legend(fontsize= '8', loc=1)
# fig.tight_layout()
# ax[1,0].set_yticks(np.arange(0.0, 0.45, 0.01)) 
# ax[1,0].set_ylim(0.41, 0.45)


# ax5 = ax[1,0].twinx()
# rects6 = ax5.bar(x+width/2, data_improve2z, width, label='Movie', color = color5)
# ax5.set_xticks(x)
# ax5.set_xticklabels(labels)
# ax5.legend(fontsize= '8', loc=2)
# fig.tight_layout()
# ax5.set_yticks(np.arange(0.0, 0.75, 0.01)) 
# ax5.set_ylim(0.71, 0.75)



# rects7 = ax[1,1].bar(x-width/2, Book_hr, width, label='Book', color = color3)
# ax[1,1].set_xticks(x)
# ax[1,1].set_xticklabels(labels)
# ax[1,1].set_xlabel('(d)')
# ax[1,1].set_title("Book-Music HR@10")
# ax[1,1].legend(fontsize= '8', loc=1)
# fig.tight_layout()
# ax[1,1].set_yticks(np.arange(0.0, 0.43, 0.01)) 
# ax[1,1].set_ylim(0.39, 0.43)


# ax6 = ax[1,1].twinx()
# rects8 = ax6.bar(x+width/2, Music_hr, width, label='Music', color = color5)
# ax6.set_xticks(x)
# ax6.set_xticklabels(labels)
# ax6.legend(fontsize= '8', loc=2)
# fig.tight_layout()
# ax6.set_yticks(np.arange(0.0, 0.755, 0.01)) 
# ax6.set_ylim(0.37, 0.41)




