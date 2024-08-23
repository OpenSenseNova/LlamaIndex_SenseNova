# LLamaIndex 集成日日新（SenseNova）
项目通过自定义LLM 的方式，使得 LlamaIndex 能够支持商汤日日新大模型（SenseNova）系列的大语言模型。提供一个简单的 LlamaIndex使用案例，包括从数据源建立索引，索引存储、从索引查询内容的完整流程。
## LLamaIndex
[LlamaIndex](https://www.llamaindex.ai/) 是一个强大的数据框架,旨在简化大型语言模型(LLMs)与外部数据的连接。它提供了一套工具和接口,使开发人员能够轻松地将自定义数据源集成到 LLM 应用中。LlamaIndex 支持各种数据类型的索引和查询,包括文本、结构化数据和even代码。它的核心功能包括数据加载、索引构建、查询接口和响应合成,这使得构建复杂的基于 LLM 的应用变得更加简单和高效。

## 商汤日日新大模型
[商汤日日新大模型开放平台](https://platform.sensenova.cn/home)是一个面向开发者的大模型开发平台的介绍。平台提供高性价比和灵活易用的解决方案，包括前沿模型、算法、推理引擎和知识库等套件，旨在帮助开发者快速构建行业模型和应用。平台的主要优势包括开放灵活、完整高效、经济适用和安全可信。其核心功能涵盖公开模型库、模型开发工具和模型应用构建。平台适用于多种场景，如智能问答、个性化推荐、内容生成和多模态应用，可应用于电商、社交、金融、短视频、广告、自动驾驶等多个行业。

## 安装
需要安装基础 python 环境，此处默认已经安装好。
### LlamaIndex 安装
```shell
pip install llama-index
```

### SenseNova 安装
```shell
pip install --upgrade sensenova
```

## 使用
### 访问密钥设置
建议通过环境变量的方式设置 AK,SK。
```shell
export SENSENOVA_ACCESS_KEY_ID=
export SENSENOVA_SECRET_ACCESS_KEY=
```
当然，也可以通过代码
```python
llm = SenseNova(model="SenseChat", access_key_id=access_key_id, secret_access_key=secret_access_key)
```

### 执行测试脚本
`data` 目录下是测试数据，  `starter.py`是测试脚本，通过以下命令执行测试：
```python
python starter.py
```
正常情况下，应该会有类似输出：
```text
...
DEBUG:urllib3.connectionpool:https://api.sensenova.cn:443 "POST /v1/llm/chat-completions HTTP/11" 200 277
https://api.sensenova.cn:443 "POST /v1/llm/chat-completions HTTP/11" 200 277
DEBUG:sensenova:message='Sensenova API response'. path=https://api.sensenova.cn/v1/llm/chat-completions. response_code=200.
message='Sensenova API response'. path=https://api.sensenova.cn/v1/llm/chat-completions. response_code=200.
The author attempted to write programs.
```

## 参考链接
- [LlamaIndex](https://www.llamaindex.ai/) 

- [SenseChat API](https://console.sensecore.cn/help/docs/model-as-a-service/nova/chat/ChatCompletions/ChatCompletion)

- [商汤日日新大模型开放平台](https://platform.sensenova.cn/home)
- [LlamaIndex Github](https://github.com/run-llama/llama_index/tree/main)