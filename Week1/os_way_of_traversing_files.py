"""os import file, this is a useful way to navigate different file images for training automation"""
#%%
import os

path = 'images/'
print(os.listdr(path))

for img in os.listdir(path):
    print(img)

# %%
"""os.walk"""
import os
print(os.walk(".")) #A Generator object

for root, dirs, files in os.walk("."):
    #print(root) #This prints out the folder contained in the current directory,root

    path = root.split(os.sep) #This is used to extract the path
    #print(path) #Spliting the root's / and then seperating . and name of files
    print(files) #This prints out all the files contains inside the folders in current directory

    #This allows us to visualize directories nad files within them
    print((len(path) - 1) * '---',os.path.basename(root)) #If depth is 2 we get ------,1 is --- the depth of
    #folders
    for file in files:
        print(len(path) * '---',file)
#%%
import os
for root,dirs,files in os.walk(root):
    #for name in dirs:
        #print(os.path.join(root,name))
    for name in files:
        print(os.path.join(root,name)) #This traverse down into the deepest file of current directory
#%%