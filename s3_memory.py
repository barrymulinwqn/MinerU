import asyncio
import io
import json
import logging
from typing import Dict, Any, List

import aioboto3
import torch
from mineru import MinerU
from sentence_transformers import SentenceTransformer

# --------------------------
# 配置（根据你的环境修改）
# --------------------------
AWS_CONFIG = {
    "aws_access_key_id": "你的AK",
    "aws_secret_access_key": "你的SK",
    "region_name": "你的区域",
    "endpoint_url": "https://s3.xxx.amazonaws.com",  # 兼容S3协议即可
}
S3_INPUT_BUCKET = "input-bucket"  # 源PDF桶
S3_OUTPUT_BUCKET = "output-bucket"  # 向量结果桶
GPU_ID = 0  # 使用的GPU
BATCH_SIZE = 8  # 根据显存调整
MAX_CONCURRENT_TASKS = 16  # 并发数（内存200G可适当加大）

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# 全局初始化（只加载一次）
# --------------------------
# GPU Embedding 模型
device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
emb_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# MinerU PDF 提取（内存模式，不写本地）
mineru = MinerU(
    model_name="MinerU-Lite",
    device=device,
    use_cache=False,  # 禁用本地缓存
)


# --------------------------
# 核心：全内存流程
# --------------------------
async def process_single_pdf(s3_key: str, s3_client) -> Dict[str, Any]:
    """
    单文件全链路：S3下载 → 内存 → PDF抽取 → GPU Embedding → 内存 → 上传S3
    全程不落盘
    """
    try:
        # 1. 从S3下载PDF【直接到内存 BytesIO，不写本地】
        pdf_bytes_io = io.BytesIO()
        await s3_client.download_fileobj(
            Bucket=S3_INPUT_BUCKET, Key=s3_key, Fileobj=pdf_bytes_io
        )
        pdf_bytes_io.seek(0)
        logger.info(f"已加载到内存: {s3_key}")

        # 2. MinerU 内存抽取文本（直接读字节流，不落地）
        extract_result = mineru.extract(
            input_data=pdf_bytes_io,  # 传入内存字节流，不是文件路径
            input_type="pdf",
            output_type="text",
        )
        text = extract_result.get("text", "").strip()
        if not text:
            logger.warning(f"无文本内容: {s3_key}")
            return {"s3_key": s3_key, "status": "empty"}

        # 3. GPU 生成 Embedding（内存→GPU，无IO）
        embedding = (
            emb_model.encode(text, convert_to_numpy=False, device=device).cpu().tolist()
        )  # 转回CPU内存

        # 4. 结果序列化到内存 BytesIO
        result_data = {
            "s3_key": s3_key,
            "text_length": len(text),
            "embedding_dim": len(embedding),
            "embedding": embedding,
        }
        result_bytes_io = io.BytesIO()
        result_bytes_io.write(
            json.dumps(result_data, ensure_ascii=False).encode("utf-8")
        )
        result_bytes_io.seek(0)

        # 5. 内存直接上传S3，不写本地
        output_key = f"embeddings/{s3_key.rsplit('.', 1)[0]}.json"
        await s3_client.upload_fileobj(
            Fileobj=result_bytes_io, Bucket=S3_OUTPUT_BUCKET, Key=output_key
        )

        logger.info(f"处理完成: {s3_key} → {output_key}")
        return {"s3_key": s3_key, "status": "success", "output_key": output_key}

    except Exception as e:
        logger.error(f"处理失败 {s3_key}: {str(e)}")
        return {"s3_key": s3_key, "status": "failed", "error": str(e)}


# --------------------------
# 批量并发调度
# --------------------------
async def batch_process(s3_keys: List[str]):
    """批量处理，控制并发，避免内存爆炸"""
    session = aioboto3.Session()
    async with session.client("s3", **AWS_CONFIG) as s3_client:
        # 信号量限制并发
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        async def safe_process(key):
            async with semaphore:
                return await process_single_pdf(key, s3_client)

        tasks = [safe_process(key) for key in s3_keys]
        results = await asyncio.gather(*tasks)
        return results


# --------------------------
# 运行入口
# --------------------------
if __name__ == "__main__":
    # 你要处理的S3文件列表
    S3_KEYS = [
        "doc/2025_report.pdf",
        "doc/manual_v1.pdf",
        # 批量添加...
    ]

    # 启动异步任务
    loop = asyncio.get_event_loop()
    final_results = loop.run_until_complete(batch_process(S3_KEYS))

    # 统计
    success = sum(1 for r in final_results if r["status"] == "success")
    empty = sum(1 for r in final_results if r["status"] == "empty")
    failed = sum(1 for r in final_results if r["status"] == "failed")
    logger.info(f"批量完成 | 成功:{success} 空文件:{empty} 失败:{failed}")
