from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # 修正：增加跨域支持
from app.api.endpoints import router # 预留路由拆分

app = FastAPI(
    title="misheng (弥生文创) API",
    description="专注于文创产品的 AI 自动化视觉渲染与宣传图生成引擎",
    version="0.1.0",
    docs_url="/docs",  # 默认文档地址
    redoc_url="/redoc" # 备用文档地址
)

# 增加 CORS 配置（为了让你未来的前端页面能顺利调用后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
