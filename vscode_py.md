# VS Code配置远程python开发环境

## 背景

最近需要在GPU，Ascend服务器上运行或调试mindspore的训练脚本。刚开始时使用Pycharm的Deployment功能把代码放在本地，每次修改代码自动上传至服务器，然后打开terminal执行程序。最大的问题是没法使用服务器上的conda环境，实现自动补全，符号跳转等功能，用起来和编辑器一样，效率很低。vscode很方便的解决了上面的问题。

## 配置远程服务器

安装[vscode](https://code.visualstudio.com/)，以及 Remote Development 插件，根据插件本身的说明，添加一个ssh remote target，配置好host name 和 user name，就可以连接到远程服务器了，支持直接访问服务器的文件目录，用起来和本地目录一样，自动同步。打开你的的代码文件夹，然后进入下一步。

## 配置python环境

第一步容易踩坑，必须在远程服务器而不是本地机器上安装 [python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)，它支持使用当前的解释器实现 code completion 和 IntelliSense，没有安装该插件会发现后续操作没有反应。确认安装成功后，按下`Ctrl+shift+P`，输入`Python: Select Interpreter` 会展示出可选的解释器，选择需要的python解释器后，就可以自由跳转，代码补全了。

## 调试运行

可以使用自带的terminal执行程序，也可以配置launch.json文件，具体参考[官方文档](https://code.visualstudio.com/docs/languages/python#_debugging)

---
参考文档
1. https://code.visualstudio.com/docs/languages/python
2. https://code.visualstudio.com/docs/remote/ssh
