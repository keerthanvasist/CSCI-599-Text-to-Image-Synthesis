Last login: Wed Nov  8 23:25:16 on ttys009
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git branch
  master
* test
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git branch master
fatal: A branch named 'master' already exists.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout master
M	GAN.py
M	create_dataset.py
Switched to branch 'master'
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   GAN.py
	modified:   create_dataset.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

no changes added to commit (use "git add" and/or "git commit -a")
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			README.md		sentences.txt
GAN.py			create_dataset.py
MNIST-100.ipynb		dataset
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add GAN.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add create_dataset.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m 
error: switch `m' requires a value
usage: git commit [<options>] [--] <pathspec>...

    -q, --quiet           suppress summary after successful commit
    -v, --verbose         show diff in commit message template

Commit message options
    -F, --file <file>     read message from file
    --author <author>     override author for commit
    --date <date>         override date for commit
    -m, --message <message>
                          commit message
    -c, --reedit-message <commit>
                          reuse and edit message from specified commit
    -C, --reuse-message <commit>
                          reuse message from specified commit
    --fixup <commit>      use autosquash formatted message to fixup specified commit
    --squash <commit>     use autosquash formatted message to squash specified commit
    --reset-author        the commit is authored by me now (used with -C/-c/--amend)
    -s, --signoff         add Signed-off-by:
    -t, --template <file>
                          use specified template file
    -e, --edit            force edit of commit
    --cleanup <default>   how to strip spaces and #comments from message
    --status              include status in commit message template
    -S, --gpg-sign[=<key-id>]
                          GPG sign commit

Commit contents options
    -a, --all             commit all changed files
    -i, --include         add specified files to index for commit
    --interactive         interactively add files
    -p, --patch           interactively add changes
    -o, --only            commit only specified files
    -n, --no-verify       bypass pre-commit and commit-msg hooks
    --dry-run             show what would be committed
    --short               show status concisely
    --branch              show branch information
    --porcelain           machine-readable output
    --long                show status in long format (default)
    -z, --null            terminate entries with NUL
    --amend               amend previous commit
    --no-post-rewrite     bypass post-rewrite hook
    -u, --untracked-files[=<mode>]
                          show untracked files, optional modes: all, normal, no. (Default: all)

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m "updating codes"
[master 3012377] updating codes
 Committer: Madhvi Kannan <madhvikannan@Madhvis-MacBook-Pro.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 2 files changed, 5 insertions(+), 43 deletions(-)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push
Counting objects: 8, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (8/8), done.
Writing objects: 100% (8/8), 4.64 KiB | 4.64 MiB/s, done.
Total 8 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
 ! [remote rejected] master -> master (permission denied)
error: failed to push some refs to 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git remote -v
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (fetch)
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (push)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (fetch)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (push)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Counting objects: 8, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (8/8), done.
Writing objects: 100% (8/8), 4.64 KiB | 4.64 MiB/s, done.
Total 8 (delta 3), reused 0 (delta 0)
remote: Resolving deltas: 100% (3/3), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
 ! [remote rejected] master -> master (permission denied)
error: failed to push some refs to 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 2), reused 3 (delta 2), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
   19cfd9f..c1b2d39  master     -> upstream/master
You asked to pull from the remote 'upstream', but did not specify
a branch. Because this is not the default configured remote
for your current branch, you must specify a branch on the command line.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Auto-merging GAN.py
CONFLICT (content): Merge conflict in GAN.py
Automatic merge failed; fix conflicts and then commit the result.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits.
  (use "git push" to publish your local commits)
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

	both modified:   GAN.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

no changes added to commit (use "git add" and/or "git commit -a")
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --cc GAN.py
index 8122170,6950038..0000000
--- a/GAN.py
+++ b/GAN.py
@@@ -3,8 -11,53 +3,58 @@@ import tensorflow as t
  import numpy as np
  import matplotlib.pyplot as plt
  import math
++<<<<<<< HEAD
 +
 + 
++=======
+ import os
+ 
+ # Load cifar-10 data
+ 
+ 
+ def load_input_images():
+     working_dir = "/dataset/input/"
+     file_list = []
+     for root, dirs, files in os.walk(working_dir):
+         
+         for filename in files:
+             if filename.endswith('.jpg'):
+                 file_list.append(root + "/" + filename) 
+     X_data=[]
+     print(file_list)
+     X_data_train=[]
+     X_data_eval=[]
+     X_data_test=[]
+     for myfile in file_list:
+         image=imread(myfile)
+         X_data.append(image)
+     X_data=np.asarray(X_data)
+     """label_list=get_label_list()
+     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
+     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
+     X_train, X_val, labels_train, labels_val=train_test_split(X_train, labels_train,test_size=0.2, random_state=42)
+     
+     X_data_train=X_train.reshape(X_train.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_eval=X_val.reshape(X_val.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_test=X_test.reshape(X_test.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+        
+     print('X_data_train shape:', np.array(X_data_train).shape)
+     print('X_data_eval shape:', np.array(X_data_eval).shape)
+     print('X_data_test shape:', np.array(X_data_test).shape)
+     print labels_train.shape
+     print labels_val.shape
+     print labels_test.shape"""
+     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
+ 
+ 
+ load_input_images()
+ #train_samples = load_input_images() / 255.0
+ #test_samples = load_test_data() / 255.0
+ 
++>>>>>>> c1b2d39e62351d232fb57fb9b9448677b938a79d
  def viz_grid(Xs, padding):
      N, H, W, C = Xs.shape
      grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ q
-bash: q: command not found
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --cc GAN.py
index 8122170,6950038..0000000
--- a/GAN.py
+++ b/GAN.py
@@@ -3,8 -11,53 +3,58 @@@ import tensorflow as t
  import numpy as np
  import matplotlib.pyplot as plt
  import math
++<<<<<<< HEAD
 +
 + 
++=======
+ import os
+ 
+ # Load cifar-10 data
+ 
+ 
+ def load_input_images():
+     working_dir = "/dataset/input/"
+     file_list = []
+     for root, dirs, files in os.walk(working_dir):
+         
+         for filename in files:
+             if filename.endswith('.jpg'):
+                 file_list.append(root + "/" + filename) 
+     X_data=[]
+     print(file_list)
+     X_data_train=[]
+     X_data_eval=[]
+     X_data_test=[]
+     for myfile in file_list:
+         image=imread(myfile)
+         X_data.append(image)
+     X_data=np.asarray(X_data)
+     """label_list=get_label_list()
+     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
+     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
+     X_train, X_val, labels_train, labels_val=train_test_split(X_train, labels_train,test_size=0.2, random_state=42)
+     
+     X_data_train=X_train.reshape(X_train.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_eval=X_val.reshape(X_val.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_test=X_test.reshape(X_test.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+        
+     print('X_data_train shape:', np.array(X_data_train).shape)
+     print('X_data_eval shape:', np.array(X_data_eval).shape)
+     print('X_data_test shape:', np.array(X_data_test).shape)
+     print labels_train.shape
+     print labels_val.shape
+     print labels_test.shape"""
+     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
+ 
+ 
+ load_input_images()
+ #train_samples = load_input_images() / 255.0
+ #test_samples = load_test_data() / 255.0
+ 
++>>>>>>> c1b2d39e62351d232fb57fb9b9448677b938a79d
  def viz_grid(Xs, padding):
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --cc GAN.py
index 8122170,6950038..0000000
--- a/GAN.py
+++ b/GAN.py
@@@ -3,8 -11,53 +3,55 @@@ import tensorflow as t
  import numpy as np
  import matplotlib.pyplot as plt
  import math
 +
 + 
+ import os
+ 
+ # Load cifar-10 data
+ 
+ 
+ def load_input_images():
+     working_dir = "/dataset/input/"
+     file_list = []
+     for root, dirs, files in os.walk(working_dir):
+         
+         for filename in files:
+             if filename.endswith('.jpg'):
+                 file_list.append(root + "/" + filename) 
+     X_data=[]
+     print(file_list)
+     X_data_train=[]
+     X_data_eval=[]
+     X_data_test=[]
+     for myfile in file_list:
+         image=imread(myfile)
+         X_data.append(image)
+     X_data=np.asarray(X_data)
+     """label_list=get_label_list()
+     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
+     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
+     X_train, X_val, labels_train, labels_val=train_test_split(X_train, labels_train,test_size=0.2, random_state=42)
+     
+     X_data_train=X_train.reshape(X_train.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_eval=X_val.reshape(X_val.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_test=X_test.reshape(X_test.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+        
+     print('X_data_train shape:', np.array(X_data_train).shape)
+     print('X_data_eval shape:', np.array(X_data_eval).shape)
+     print('X_data_test shape:', np.array(X_data_test).shape)
+     print labels_train.shape
+     print labels_val.shape
+     print labels_test.shape"""
+     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
+ 
+ 
+ load_input_images()
+ #train_samples = load_input_images() / 255.0
+ #test_samples = load_test_data() / 255.0
+ 
  def viz_grid(Xs, padding):
      N, H, W, C = Xs.shape
      grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream
error: Pulling is not possible because you have unmerged files.
hint: Fix them up in the work tree, and then use 'git add/rm <file>'
hint: as appropriate to mark resolution and make a commit.
fatal: Exiting because of an unresolved conflict.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits.
  (use "git push" to publish your local commits)
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

	both modified:   GAN.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

no changes added to commit (use "git add" and/or "git commit -a")
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --cc GAN.py
index 8122170,6950038..0000000
--- a/GAN.py
+++ b/GAN.py
@@@ -3,8 -11,53 +3,55 @@@ import tensorflow as t
  import numpy as np
  import matplotlib.pyplot as plt
  import math
 +
 + 
+ import os
+ 
+ # Load cifar-10 data
+ 
+ 
+ def load_input_images():
+     working_dir = "/dataset/input/"
+     file_list = []
+     for root, dirs, files in os.walk(working_dir):
+         
+         for filename in files:
+             if filename.endswith('.jpg'):
+                 file_list.append(root + "/" + filename) 
+     X_data=[]
+     print(file_list)
+     X_data_train=[]
+     X_data_eval=[]
+     X_data_test=[]
+     for myfile in file_list:
+         image=imread(myfile)
+         X_data.append(image)
+     X_data=np.asarray(X_data)
+     """label_list=get_label_list()
+     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
+     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
+     X_train, X_val, labels_train, labels_val=train_test_split(X_train, labels_train,test_size=0.2, random_state=42)
+     
+     X_data_train=X_train.reshape(X_train.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_eval=X_val.reshape(X_val.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_test=X_test.reshape(X_test.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+        
+     print('X_data_train shape:', np.array(X_data_train).shape)
+     print('X_data_eval shape:', np.array(X_data_eval).shape)
+     print('X_data_test shape:', np.array(X_data_test).shape)
+     print labels_train.shape
+     print labels_val.shape
+     print labels_test.shape"""
+     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
+ 
+ 
+ load_input_images()
+ #train_samples = load_input_images() / 255.0
+ #test_samples = load_test_data() / 255.0
+ 
  def viz_grid(Xs, padding):
      N, H, W, C = Xs.shape
      grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge
error: Merging is not possible because you have unmerged files.
hint: Fix them up in the work tree, and then use 'git add/rm <file>'
hint: as appropriate to mark resolution and make a commit.
fatal: Exiting because of an unresolved conflict.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge upstream
error: Merging is not possible because you have unmerged files.
hint: Fix them up in the work tree, and then use 'git add/rm <file>'
hint: as appropriate to mark resolution and make a commit.
fatal: Exiting because of an unresolved conflict.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi create_dataset.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --cc GAN.py
index 8122170,6950038..0000000
--- a/GAN.py
+++ b/GAN.py
@@@ -3,8 -11,53 +3,55 @@@ import tensorflow as t
  import numpy as np
  import matplotlib.pyplot as plt
  import math
 +
 + 
+ import os
+ 
+ # Load cifar-10 data
+ 
+ 
+ def load_input_images():
+     working_dir = "/dataset/input/"
+     file_list = []
+     for root, dirs, files in os.walk(working_dir):
+         
+         for filename in files:
+             if filename.endswith('.jpg'):
+                 file_list.append(root + "/" + filename) 
+     X_data=[]
+     print(file_list)
+     X_data_train=[]
+     X_data_eval=[]
+     X_data_test=[]
+     for myfile in file_list:
+         image=imread(myfile)
+         X_data.append(image)
+     X_data=np.asarray(X_data)
+     """label_list=get_label_list()
+     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
+     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
+     X_train, X_val, labels_train, labels_val=train_test_split(X_train, labels_train,test_size=0.2, random_state=42)
+     
+     X_data_train=X_train.reshape(X_train.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_eval=X_val.reshape(X_val.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+     X_data_test=X_test.reshape(X_test.shape[0], 1, 28, 28).transpose(
+         0, 2, 3, 1).astype("uint8")
+        
+     print('X_data_train shape:', np.array(X_data_train).shape)
+     print('X_data_eval shape:', np.array(X_data_eval).shape)
+     print('X_data_test shape:', np.array(X_data_test).shape)
+     print labels_train.shape
+     print labels_val.shape
+     print labels_test.shape"""
+     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
+ 
+ 
+ load_input_images()
+ #train_samples = load_input_images() / 255.0
+ #test_samples = load_test_data() / 255.0
+ 
  def viz_grid(Xs, padding):
      N, H, W, C = Xs.shape
      grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$  git pull
error: Pulling is not possible because you have unmerged files.
hint: Fix them up in the work tree, and then use 'git add/rm <file>'
hint: as appropriate to mark resolution and make a commit.
fatal: Exiting because of an unresolved conflict.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is ahead of 'origin/master' by 3 commits.
  (use "git push" to publish your local commits)
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

	both modified:   GAN.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

no changes added to commit (use "git add" and/or "git commit -a")
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m "Resolve merge conflict"
[master ed5107d] Resolve merge conflict
 Committer: Madhvi Kannan <madhvikannan@Madhvis-MacBook-Pro.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly. Run the
following command and follow the instructions in your editor to edit
your configuration file:

    git config --global --edit

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Counting objects: 14, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (14/14), done.
Writing objects: 100% (14/14), 5.72 KiB | 2.86 MiB/s, done.
Total 14 (delta 7), reused 0 (delta 0)
remote: Resolving deltas: 100% (7/7), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
 ! [remote rejected] master -> master (permission denied)
error: failed to push some refs to 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream
You asked to pull from the remote 'upstream', but did not specify
a branch. Because this is not the default configured remote
for your current branch, you must specify a branch on the command line.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is ahead of 'origin/master' by 5 commits.
  (use "git push" to publish your local commits)
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			GAN.py			MNIST-100.ipynb		README.md		create_dataset.py	dataset			sentences.txt
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git branch test1
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout test1
Switched to branch 'test1'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test1
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin
fatal: The current branch test1 has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin test1

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git remote -v
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (fetch)
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (push)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (fetch)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (push)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Counting objects: 14, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (14/14), done.
Writing objects: 100% (14/14), 5.72 KiB | 2.86 MiB/s, done.
Total 14 (delta 7), reused 0 (delta 0)
remote: Resolving deltas: 100% (7/7), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
 ! [remote rejected] master -> master (permission denied)
error: failed to push some refs to 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git branch
  master
  test
* test1
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git config --global --edit
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Username for 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': mkannan
Password for 'https://mkannan@github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': 
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git/'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Username for 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': mkannan
Password for 'https://mkannan@github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': 
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git/'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Username for 'https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': MadhviKannan
Password for 'https://MadhviKannan@github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git': 
Counting objects: 14, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (14/14), done.
Writing objects: 100% (14/14), 5.72 KiB | 1.43 MiB/s, done.
Total 14 (delta 7), reused 0 (delta 0)
remote: Resolving deltas: 100% (7/7), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
   638fde2..ed5107d  master -> master
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 3 (delta 2), reused 3 (delta 2), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
   c1b2d39..747207d  master     -> upstream/master
You asked to pull from the remote 'upstream', but did not specify
a branch. Because this is not the default configured remote
for your current branch, you must specify a branch on the command line.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
error: Your local changes to the following files would be overwritten by merge:
	GAN.py
Please commit your changes or stash them before you merge.
Aborting
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --git a/GAN.py b/GAN.py
index a8b76d5..cfff727 100644
--- a/GAN.py
+++ b/GAN.py
@@ -1,17 +1,17 @@
 # Import required libraries
-import tensorflow as tf
+#mport tensorflow as tf
 import numpy as np
 import matplotlib.pyplot as plt
 import math
-
- 
+import sklearn
+from scipy.ndimage import imread
 import os
 
 # Load cifar-10 data
 
 
 def load_input_images():
-    working_dir = "/dataset/input/"
+    working_dir = "./dataset/input"
     file_list = []
     for root, dirs, files in os.walk(working_dir):
         
@@ -27,6 +27,7 @@ def load_input_images():
         image=imread(myfile)
         X_data.append(image)
     X_data=np.asarray(X_data)
+    
     """label_list=get_label_list()
     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
@@ -48,10 +49,25 @@ def load_input_images():
     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
 
 
-load_input_images()
+#load_input_images()
 #train_samples = load_input_images() / 255.0
 #test_samples = load_test_data() / 255.0
-
+def load_input_sentences():
+    sentence_file=open('./dataset/input/sentences.txt','r')
+    sentence=sentence_file.readline()
+    
+    sentences=[]
+    solutions=[]
+    
+    while sentence!="":
+        sentences.append(sentence)
+        solution=sentence.split(' ')[-1]
+        solutions.append(int(solution))
+        sentence=sentence_file.readline()
+    print sentences
+    return sentences, solutions
+    
+input_sentences, input_solutions=load_input_sentences()
 def viz_grid(Xs, padding):
     N, H, W, C = Xs.shape
     grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge GAN.py 
merge: GAN.py - not something we can merge
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --git a/GAN.py b/GAN.py
index a8b76d5..cfff727 100644
--- a/GAN.py
+++ b/GAN.py
@@ -1,17 +1,17 @@
 # Import required libraries
-import tensorflow as tf
+#mport tensorflow as tf
 import numpy as np
 import matplotlib.pyplot as plt
 import math
-
- 
+import sklearn
+from scipy.ndimage import imread
 import os
 
 # Load cifar-10 data
 
 
 def load_input_images():
-    working_dir = "/dataset/input/"
+    working_dir = "./dataset/input"
     file_list = []
     for root, dirs, files in os.walk(working_dir):
         
@@ -27,6 +27,7 @@ def load_input_images():
         image=imread(myfile)
         X_data.append(image)
     X_data=np.asarray(X_data)
+    
     """label_list=get_label_list()
     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
@@ -48,10 +49,25 @@ def load_input_images():
     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
 
 
-load_input_images()
+#load_input_images()
 #train_samples = load_input_images() / 255.0
 #test_samples = load_test_data() / 255.0
-
+def load_input_sentences():
+    sentence_file=open('./dataset/input/sentences.txt','r')
+    sentence=sentence_file.readline()
+    
+    sentences=[]
+    solutions=[]
+    
+    while sentence!="":
+        sentences.append(sentence)
+        solution=sentence.split(' ')[-1]
+        solutions.append(int(solution))
+        sentence=sentence_file.readline()
+    print sentences
+    return sentences, solutions
+    
+input_sentences, input_solutions=load_input_sentences()
 def viz_grid(Xs, padding):
     N, H, W, C = Xs.shape
     grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git fetch upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi diff
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git diff
diff --git a/GAN.py b/GAN.py
index a8b76d5..cfff727 100644
--- a/GAN.py
+++ b/GAN.py
@@ -1,17 +1,17 @@
 # Import required libraries
-import tensorflow as tf
+#mport tensorflow as tf
 import numpy as np
 import matplotlib.pyplot as plt
 import math
-
- 
+import sklearn
+from scipy.ndimage import imread
 import os
 
 # Load cifar-10 data
 
 
 def load_input_images():
-    working_dir = "/dataset/input/"
+    working_dir = "./dataset/input"
     file_list = []
     for root, dirs, files in os.walk(working_dir):
         
@@ -27,6 +27,7 @@ def load_input_images():
         image=imread(myfile)
         X_data.append(image)
     X_data=np.asarray(X_data)
+    
     """label_list=get_label_list()
     X_data_shuffle, label_list_shuffle=X_data.reshape(60000,1,28,28), label_list
     X_train, X_test, labels_train, labels_test=train_test_split(X_data_shuffle, label_list_shuffle,test_size=0.2, random_state=42)
@@ -48,10 +49,25 @@ def load_input_images():
     #return X_data_train, labels_train, X_data_eval, labels_val, X_data_test, labels_test
 
 
-load_input_images()
+#load_input_images()
 #train_samples = load_input_images() / 255.0
 #test_samples = load_test_data() / 255.0
-
+def load_input_sentences():
+    sentence_file=open('./dataset/input/sentences.txt','r')
+    sentence=sentence_file.readline()
+    
+    sentences=[]
+    solutions=[]
+    
+    while sentence!="":
+        sentences.append(sentence)
+        solution=sentence.split(' ')[-1]
+        solutions.append(int(solution))
+        sentence=sentence_file.readline()
+    print sentences
+    return sentences, solutions
+    
+input_sentences, input_solutions=load_input_sentences()
 def viz_grid(Xs, padding):
     N, H, W, C = Xs.shape
     grid_size = int(math.ceil(math.sqrt(N)))
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git stash
Saved working directory and index state WIP on test1: ed5107d Resolve merge conflict
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details.

    git pull <remote> <branch>

If you wish to set tracking information for this branch you can do so with:

    git branch --set-upstream-to=<remote>/<branch> test1

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Auto-merging GAN.py
Merge made by the 'recursive' strategy.
 GAN.py | 60 +++++++++++++++++++++++++++++++-----------------------------
 1 file changed, 31 insertions(+), 29 deletions(-)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test1
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Everything up-to-date
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m "GAN"
On branch test1
Untracked files:
	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Everything up-to-date
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test1
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout master
Switched to branch 'master'
Your branch is up-to-date with 'origin/master'.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	Data/
	dataset/

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout test1
Switched to branch 'test1'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			GAN.py			MNIST-100.ipynb		README.md		create_dataset.py	dataset			sentences.txt
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push
fatal: The current branch test1 has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin test1

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git remote -v
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (fetch)
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (push)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (fetch)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (push)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Everything up-to-date
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			MNIST-100.ipynb		create_dataset.py	sentences.txt		skipthoughts.pyc
GAN.py			README.md		dataset			skipthoughts.py		test.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test2
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   test.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   GAN.py
	modified:   test.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	.idea/
	Data/
	dataset/
	skipthoughts.py
	skipthoughts.pyc

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m "Added skip thought vector"
[test2 4860f5b] Added skip thought vector
 Committer: mkannan@usc.edu <madhvikannan@Madhvis-MacBook-Pro.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 2 files changed, 2 insertions(+)
 create mode 100644 test.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Counting objects: 12, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (12/12), done.
Writing objects: 100% (12/12), 1.83 KiB | 935.00 KiB/s, done.
Total 12 (delta 8), reused 0 (delta 0)
remote: Resolving deltas: 100% (8/8), completed with 2 local objects.
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
   ed5107d..b75631e  master -> master
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin test
Total 0 (delta 0), reused 0 (delta 0)
To https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git
 * [new branch]      test -> test
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			MNIST-100.ipynb		create_dataset.py	sentences.txt		skipthoughts.pyc
GAN.py			README.md		dataset			skipthoughts.py		test.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test2
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   test.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	.idea/
	Data/
	dataset/
	skipthoughts.py
	skipthoughts.pyc

no changes added to commit (use "git add" and/or "git commit -a")
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git add skipthoughts.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git commit -m "skip thoughts from git"
[test2 3fc8ce4] skip thoughts from git
 Committer: mkannan@usc.edu <madhvikannan@Madhvis-MacBook-Pro.local>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 1 file changed, 438 insertions(+)
 create mode 100644 skipthoughts.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push
fatal: The current branch test2 has no upstream branch.
To push the current branch and set the remote as upstream, use

    git push --set-upstream origin test2

Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge master
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout master
error: Your local changes to the following files would be overwritten by checkout:
	test.py
Please commit your changes or stash them before you switch branches.
Aborting
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git stash
Saved working directory and index state WIP on test2: 3fc8ce4 skip thoughts from git
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout master
Switched to branch 'master'
Your branch is up-to-date with 'origin/master'.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git remote -v
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (fetch)
origin	https://github.com/MadhviKannan/CSCI-599-Text-to-Image-Synthesis.git (push)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (fetch)
upstream	https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git (push)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout test
Switched to branch 'test'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git branch
  master
* test
  test1
  test2
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout test2
Switched to branch 'test2'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge master test
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git merge test master
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push origin master
Everything up-to-date
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ ls
Data			MNIST-100.ipynb		create_dataset.py	sentences.txt		skipthoughts.pyc
GAN.py			README.md		dataset			skipthoughts.py		test.py
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git status
On branch test2
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	.DS_Store
	.idea/
	Data/
	dataset/
	skipthoughts.pyc

nothing added to commit but untracked files present (use "git add" to track)
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git push upstream master
Username for 'https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git': keerthanvasist
Password for 'https://keerthanvasist@github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git': 
Counting objects: 12, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (12/12), done.
Writing objects: 100% (12/12), 2.01 KiB | 1.01 MiB/s, done.
Total 12 (delta 7), reused 0 (delta 0)
remote: Resolving deltas: 100% (7/7), completed with 4 local objects.
To https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis.git
   8f8af2d..b75631e  master -> master
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout master
Switched to branch 'master'
Your branch is up-to-date with 'origin/master'.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git pull upstream master
From https://github.com/keerthanvasist/CSCI-599-Text-to-Image-Synthesis
 * branch            master     -> FETCH_HEAD
Already up-to-date.
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ git checkout test2
Switched to branch 'test2'
Madhvis-MacBook-Pro:CSCI-599-Text-to-Image-Synthesis madhvikannan$ vi GAN.py 


            run_ops = [self.recon_loss_op, self.reconstruct_op, self.actmax_sample_op]
            last_loss, _, last_reconstruction = sess.run(run_ops, feed_dict = recon_feed_dict)
        return last_loss, last_reconstruction

    # Find the reconstruction of a batch of samples
    def reconstruct(self, samples):
        reconstructions = np.zeros(samples.shape)
        total_loss = 0
        for i in range(samples.shape[0]):
            loss, reconstructions[i:i+1] = self.reconstruct_one_sample(samples[i:i+1])
            total_loss += loss
        return total_loss / samples.shape[0], reconstructions

    # Generates a single sample from input code
    def generate_one_sample(self, code):

        ################################################################################
        # Prob 2-1: complete the feed dictionary                                       #
        ################################################################################

        gen_vis_feed_dict = {self.noise:code, self.is_train:False}

        ################################################################################
        #                               END OF YOUR CODE                               #
        ################################################################################

        generated = sess.run(self.fake_samples_op, feed_dict = gen_vis_feed_dict)
        return generated

    # Generates samples from input batch of codes
    def generate(self, codes):
        generated = np.zeros((codes.shape[0], 32, 32, 3))
        for i in range(codes.shape[0]):
            generated[i:i+1] = self.generate_one_sample(codes[i:i+1])
        return generated

    # Perform activation maximization on one initial code
    def actmax_one_sample(self, initial_code):

        ################################################################################
        # Prob 2-4: check this function                                                #
        # skip this part when working on problem 2-1 and come back for problem 2-4     #
        ################################################################################

        actmax_init_val = tf.convert_to_tensor(initial_code, dtype = tf.float32)
        sess.run(self.actmax_code.assign(actmax_init_val))
        for i in range(self.actmax_steps):
            actmax_feed_dict = {
                self.actmax_label: np.ones([1, 1]),
                self.is_train: False
            }
            _, last_actmax = sess.run([self.actmax_op, self.actmax_sample_op], feed_dict = actmax_feed_dict)
        return last_actmax

    # Perform activation maximization on a batch of different initial codes
    def actmax(self, initial_codes):
        actmax_results = np.zeros((initial_codes.shape[0], 32, 32, 3))
        for i in range(initial_codes.shape[0]):
            actmax_results[i:i+1] = self.actmax_one_sample(initial_codes[i:i+1])
        return actmax_results.clip(0, 1)

