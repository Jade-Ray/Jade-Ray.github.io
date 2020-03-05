# github pages 搭建记录

web模板来自[beautiful-jekyll](https://github.com/daattali/beautiful-jekyll)

自定义模板的参考有：

- [黑客风格模板](https://github.com/akiritsu/pRoJEct-NeGYa)

- [官方hacker主题](https://github.com/pages-themes/hacker)

主要参考文档：[jekyll](https://jekyllrb.com/)



### 知识点

---

#### prose.io

一个简单的可以在Web上编辑文件的工具，直接访问[prose.io](http://prose.io)，授权后就可以浏览仓库并修改其中文件。

可以在_config.yml里面添加一些prose.io使用的配置，参考[官方文档](https://github.com/prose/prose/wiki/Prose-Configuration)。

#### RSS feed

> RSS（简易信息聚合）是一种消息来源格式规范，用以聚合经常发布更新数据的网站，例如博客文章、新闻、音频或视频的网摘。RSS文件（或称做摘要、网络摘要、或频更新，提供到频道）包含全文或是节录的文字，再加上发布者所订阅之网摘数据和授权的元数据。

在浏览器中可以订阅RSS，Chrome插件RSS Feed Reader，RSS地址再模板中为feed.xml

#### jekyll插件

- jekyll-sitemap 自动生成站点地图，可用在GitHub Pages，[参考这里](https://help.github.com/en/github/working-with-github-pages/about-github-pages-and-jekyll)
- jekyll-paginate 分页功能插件，[参考官网](https://jekyllrb.com/docs/pagination/)
- jekyll-seo-tag 对搜索引擎添加元标签， [参考这里](https://github.com/jekyll/jekyll-seo-tag)

#### StaticMan

一个三方的评论系统，原理是提供一个 API，当用户发布评论（即提交表单时），它把这些信息按照结构化数据格式写入文件中（Jekyll中通常是`yml`文件），再把文件写入你的Github仓库中。[官网](https://staticman.net/)

