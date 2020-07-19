---
layout: post
title:  "Github cheatsheet"
date:   2020-07-19 14:55:04 +0545
categories: jekyll blog

---
Useful github cheatsheet

### Purpose of the blog:
Add empty .keep files in all directories which rae empty


### Resources
Link can be found [here at cartoon-gan paper][colab-runtime-steps].

### Prerequisites
Should be able to run unix cli command

```
##code credit http://cavaliercoder.com/blog/recursively-create-gitkeep-files.html
find . -type d -empty -not -path "./.git/*" -exec touch {}/.keep \;

```
### Visualize the directory
Visualize with hidden files

```
tree -a
```




