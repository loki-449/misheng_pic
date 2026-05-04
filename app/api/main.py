from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware # 修正：增加跨域支持
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

_WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时打印首页路径，便于排查「仍是 404」时是否连错了进程或缺文件。"""
    ok = _WEB_INDEX.is_file()
    print(f"[misheng-poster-agent] 演示页文件: {_WEB_INDEX} 存在={ok}")
    if not ok:
        print("[misheng-poster-agent] 请确认仓库里有 app/web/index.html，并重启 uvicorn。")
    yield


app = FastAPI(
    title="misheng (弥生文创) API",
    description="专注于文创产品的 AI 自动化视觉渲染与宣传图生成引擎",
    version="0.1.0",
    docs_url="/docs",  # 默认文档地址
    redoc_url="/redoc",  # 备用文档地址
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 时附带中文说明，便于网页与脚本排查 multipart 字段名。"""
    errors = exc.errors()
    message_zh = ""
    path = request.url.path or ""
    if "/poster/mvp/run" in path:
        locs = [str(e.get("loc", ())) for e in errors]
        joined = " ".join(locs).lower()
        if "product_images" in joined or "product_image" in joined:
            message_zh = (
                "上传字段校验失败：请使用 multipart 表单上传商品图。"
                "字段名应为 **product_images**（可同名重复多张），或使用旧版单字段 **product_image**。"
                "纯 JSON 或缺少文件部分会触发本错误。详见接口文档 /docs 中该接口的说明。"
            )
    body: dict = {"detail": errors}
    if message_zh:
        body["message_zh"] = message_zh
    return JSONResponse(status_code=422, content=body)


def _home_response():
    if not _WEB_INDEX.is_file():
        return PlainTextResponse(
            "找不到演示页文件：app/web/index.html\n请从项目根目录启动：uvicorn app.api.main:app --host 127.0.0.1 --port 8000",
            status_code=503,
        )
    html = _WEB_INDEX.read_text(encoding="utf-8")
    return HTMLResponse(content=html, media_type="text/html; charset=utf-8")


# 先注册首页，再挂其它路由（避免极端情况下路由表顺序带来的困惑）
@app.get("/")
async def serve_home():
    """浏览器打开 http://127.0.0.1:8000/ 应看到上传演示页。"""
    return _home_response()


@app.get("/index.html")
async def serve_home_index():
    """部分环境会请求 /index.html，与根路径一致。"""
    return _home_response()


# 增加 CORS 配置（为了让你未来的前端页面能顺利调用后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.endpoints import router  # noqa: E402 — 在首页路由之后再加载业务（含重型依赖）

# 挂载业务路由
app.include_router(router.router, prefix="/v1")

# 暴露静态目录，便于前端直接访问渲染结果与上传资源
app.mount("/static", StaticFiles(directory="app/static"), name="static")
