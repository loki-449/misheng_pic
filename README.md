# misheng-poster-agent

基于 FastAPI 的海报生成 Agent 服务，支持产品视觉方案生成与质量评审。

## Quick Start

1. 安装依赖
   - `pip install -r requirements.txt`
2. 准备配置
   - 复制 `.env.example` 为 `.env`
   - 填写 `DEEPSEEK_API_KEY`
3. 启动服务
   - `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`

## API Endpoints

- `GET /v1/health`：健康检查
- `GET /v1/config`：查看非敏感运行配置
- `POST /v1/poster/plan`：生成视觉方案
- `POST /v1/poster/review`：评估视觉方案质量
- `POST /v1/poster/pipeline`：一步完成 plan + review
- `POST /v1/poster/render`：最小版渲染出图

## cURL Examples

建议每次请求携带 `x-request-id`，便于日志追踪。

### 1) plan

```bash
curl -X POST "http://127.0.0.1:8000/v1/poster/plan" \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-plan-001" \
  -d '{
    "product_name": "樱花和纸胶带",
    "product_desc": "粉色半透明胶带，带春日意象",
    "canvas_size": 2048
  }'
```

### 2) review

```bash
curl -X POST "http://127.0.0.1:8000/v1/poster/review" \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-review-001" \
  -d '{
    "style": {"tags": ["clean"], "confidence": 0.8, "reason": "soft palette"},
    "layout": {"canvas_size": 2048, "composition": "product_center_bottom", "text_area": "top_left", "safe_margin_percent": 8},
    "background": {"mode": "retrieve_or_generate", "keywords": ["soft light"], "negative_keywords": ["busy pattern"], "note": "prioritize readability"}
  }'
```

### 3) pipeline

```bash
curl -X POST "http://127.0.0.1:8000/v1/poster/pipeline" \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-pipeline-001" \
  -d '{
    "product_name": "手账贴纸套装",
    "product_desc": "治愈系植物手账贴纸，面向年轻用户",
    "canvas_size": 2048
  }'
```

### 4) render

```bash
curl -X POST "http://127.0.0.1:8000/v1/poster/render" \
  -H "Content-Type: application/json" \
  -H "x-request-id: demo-render-001" \
  -d '{
    "product_name": "手账贴纸套装",
    "product_desc": "治愈系植物手账贴纸，面向年轻用户",
    "tagline": "新品推荐",
    "canvas_size": 2048,
    "product_image_path": "app/static/uploads/product.png"
  }'
```

### 5) render (upload file)

```bash
curl -X POST "http://127.0.0.1:8000/v1/poster/render/upload" \
  -H "x-request-id: demo-render-upload-001" \
  -F "product_name=手账贴纸套装" \
  -F "product_desc=治愈系植物手账贴纸，面向年轻用户" \
  -F "tagline=新品推荐" \
  -F "canvas_size=2048" \
  -F "product_image=@app/static/uploads/product.png"
```

上传限制：

- 文件类型：`image/png`、`image/jpeg`、`image/webp`
- 文件大小：默认不超过 `10MB`（可用 `MAX_UPLOAD_BYTES` 调整）
- 自动清理：上传与结果目录默认保留 `7` 天，最多 `500` 个文件（可用 `STORAGE_MAX_AGE_DAYS`、`STORAGE_MAX_FILES` 调整）

静态资源访问：

- 服务已挂载 `/static`，渲染返回的 `image_path` / `thumbnail_path` 可直接拼接域名访问
- 示例：`http://127.0.0.1:8000/static/results/xxx.png`

## Test

运行最小接口测试：

- `pytest -q`

使用 Docker 执行测试：

- `bash docker_test.sh`

## CI

项目已包含 GitHub Actions 工作流：`.github/workflows/ci.yml`

- 触发时机：`push main` 和 `pull_request`
- 执行内容：安装依赖并运行 `pytest -q`
