from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from skimage import io
from PIL import Image
import numpy as np
from azure.storage.blob import BlobServiceClient
from typing import Optional, Tuple
import logging
from dotenv import load_dotenv
from io import BytesIO
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Azure Blob Storage configuration
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in environment")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT", "bivlargefiles")
container_name = os.getenv("AZURE_CONTAINER", "cryovizweb")

INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
NEXT_BASE_URL = os.getenv("NEXT_BASE_URL", "http://localhost:3000")

# In-memory cancellation flags keyed by uploadId
CANCEL_FLAGS: dict[str, bool] = {}

async def post_status(session: aiohttp.ClientSession, upload_id: str, user_id: str, payload: dict):
    url = f"{NEXT_BASE_URL}/api/upload-status"
    headers = {"x-internal-secret": INTERNAL_API_SECRET or ""}
    body = {"uploadId": upload_id, "userId": user_id, **payload}
    async with session.post(url, json=body, headers=headers) as resp:
        text = await resp.text()
        if resp.status >= 300:
            logger.warning("Status post failed %s: %s", resp.status, text)

async def download_from_azure_url(azure_url: str, local_path: str):
    """Download file from Azure blob URL to local path"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(azure_url) as response:
                if response.status == 200:
                    with open(local_path, "wb") as f:
                        f.write(await response.read())
                    logger.info(f"Downloaded {azure_url} to {local_path}")
                else:
                    raise Exception(f"Failed to download from Azure: {response.status}")
    except Exception as e:
        logger.error(f"Error downloading from Azure: {e}")
        raise

async def process_tiff_from_azure(azure_url: str, filename: Optional[str], alpha_file_path: Optional[str], dataset_id: str, modality: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Process TIFF file downloaded from Azure URL"""
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_tiff_path = os.path.join(temp_dir, filename or f"{modality}_{dataset_id}.tiff")

    try:
        # Download from Azure
        await download_from_azure_url(azure_url, temp_tiff_path)
        
        # Process the downloaded file
        return await process_tiff_stack_from_file(temp_tiff_path, alpha_file_path, dataset_id, modality)
    finally:
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)

async def process_tiff_stack_from_file(tiff_path: str, alpha_file_path: Optional[str], dataset_id: str, modality: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """Process TIFF file from local path (existing logic)"""
    try:
        logger.info("Reading %s TIFF: %s", modality, tiff_path)
        rgba_stack = io.imread(tiff_path)
        if rgba_stack.ndim != 4 or rgba_stack.shape[-1] != 4:
            raise HTTPException(status_code=400, detail=f"Invalid TIFF format for {modality}: must be RGBA with shape (Z, Y, X, 4)")

        Z, Y, X, _ = rgba_stack.shape
        logger.info("%s dimensions: Z=%d, Y=%d, X=%d", modality, Z, Y, X)

        alpha_stack = None
        if alpha_file_path and os.path.exists(alpha_file_path):
            alpha_stack = io.imread(alpha_file_path)
            if alpha_stack.shape != (Z, Y, X):
                raise HTTPException(status_code=400, detail=f"Alpha mask dimensions {alpha_stack.shape} do not match {modality} stack {rgba_stack.shape}")

        alpha_threshold = 10

        def apply_alpha_mask(rgba, alpha_mask=None):
            rgba = rgba.copy()
            if alpha_mask is not None:
                mask = alpha_mask < alpha_threshold
                rgba[mask] = [0, 0, 0, 0]
            return rgba

        container_client = blob_service_client.get_container_client(container_name)
        blob_prefix = f"dataset-{dataset_id}/{modality}"

        # Save XY slices
        for z in range(Z):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_slice = rgba_stack[z]
            alpha_mask = alpha_stack[z] if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_slice, alpha_mask))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xy/{z:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save XZ slices
        for y in range(Y):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_xz = np.stack([rgba_stack[z, y, :, :] for z in range(Z)], axis=0)
            alpha_xz = np.stack([alpha_stack[z, y, :] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_xz, alpha_xz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xz/{y:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save YZ slices
        for x in range(X):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_yz = np.stack([rgba_stack[z, :, x, :] for z in range(Z)], axis=0)
            alpha_yz = np.stack([alpha_stack[z, :, x] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_yz, alpha_yz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/yz/{x:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_prefix}"
        return blob_url, Z, Y, X

    except Exception as e:
        logger.error(f"Error processing TIFF file: {e}")
        raise


async def process_tiff_stack(tiff_bytes: Optional[bytes], filename: Optional[str], alpha_file_path: Optional[str], dataset_id: str, modality: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    if not tiff_bytes:
        return None, None, None, None

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_tiff_path = os.path.join(temp_dir, filename or f"{modality}_{dataset_id}.tiff")

    try:
        with open(temp_tiff_path, "wb") as f:
            f.write(tiff_bytes)

        logger.info("Reading %s TIFF: %s", modality, temp_tiff_path)
        rgba_stack = io.imread(temp_tiff_path)
        if rgba_stack.ndim != 4 or rgba_stack.shape[-1] != 4:
            raise HTTPException(status_code=400, detail=f"Invalid TIFF format for {modality}: must be RGBA with shape (Z, Y, X, 4)")

        Z, Y, X, _ = rgba_stack.shape
        logger.info("%s dimensions: Z=%d, Y=%d, X=%d", modality, Z, Y, X)

        alpha_stack = None
        if alpha_file_path and os.path.exists(alpha_file_path):
            alpha_stack = io.imread(alpha_file_path)
            if alpha_stack.shape != (Z, Y, X):
                raise HTTPException(status_code=400, detail=f"Alpha mask dimensions {alpha_stack.shape} do not match {modality} stack {rgba_stack.shape}")

        alpha_threshold = 10

        def apply_alpha_mask(rgba, alpha_mask=None):
            rgba = rgba.copy()
            if alpha_mask is not None:
                mask = alpha_mask < alpha_threshold
                rgba[mask] = [0, 0, 0, 0]
            return rgba

        container_client = blob_service_client.get_container_client(container_name)
        blob_prefix = f"dataset-{dataset_id}/{modality}"

        # Save XY slices
        for z in range(Z):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_slice = rgba_stack[z]
            alpha_mask = alpha_stack[z] if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_slice, alpha_mask))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xy/{z:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save XZ slices
        for y in range(Y):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_xz = np.stack([rgba_stack[z, y, :, :] for z in range(Z)], axis=0)
            alpha_xz = np.stack([alpha_stack[z, y, :] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_xz, alpha_xz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xz/{y:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save YZ slices
        for x in range(X):
            if CANCEL_FLAGS.get(dataset_id) or CANCEL_FLAGS.get(modality) or CANCEL_FLAGS.get("__" + dataset_id):
                raise Exception("Cancelled")
            rgba_yz = np.stack([rgba_stack[z, :, x, :] for z in range(Z)], axis=0)
            alpha_yz = np.stack([alpha_stack[z, :, x] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_yz, alpha_yz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/yz/{x:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_prefix}"
        return blob_url, Z, Y, X

    finally:
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)


async def process_dataset_bg(
    upload_id: str,
    user_id: str,
    name: str,
    description: str,
    institution_id: str,
    spacing: str,
    bf_temp_url: Optional[str],
    bf_filename: Optional[str],
    fl_temp_url: Optional[str],
    fl_filename: Optional[str],
    alpha_temp_url: Optional[str],
):
    dataset_id = str(uuid.uuid4())
    alpha_temp_path = None

    async with aiohttp.ClientSession() as session:
        try:
            await post_status(session, upload_id, user_id, {
                "status": "processing",
                "progress": 20,
                "message": "Starting processing..."
            })

            # Download alpha from Azure if provided
            alpha_temp_path = None
            if alpha_temp_url:
                temp_dir = "./temp"
                os.makedirs(temp_dir, exist_ok=True)
                alpha_temp_path = os.path.join(temp_dir, f"alpha_{dataset_id}.tiff")
                await download_from_azure_url(alpha_temp_url, alpha_temp_path)

            brightfield_blob_url = None
            fluorescent_blob_url = None
            bfZ = bfY = bfX = None
            flZ = flY = flX = None

            if bf_temp_url:
                await post_status(session, upload_id, user_id, {
                    "status": "processing",
                    "progress": 40,
                    "message": "Processing brightfield..."
                })
                brightfield_blob_url, bfZ, bfY, bfX = await process_tiff_from_azure(bf_temp_url, bf_filename, alpha_temp_path, dataset_id, "brightfield")

            if fl_temp_url:
                await post_status(session, upload_id, user_id, {
                    "status": "processing",
                    "progress": 60,
                    "message": "Processing fluorescent..."
                })
                fluorescent_blob_url, flZ, flY, flX = await process_tiff_from_azure(fl_temp_url, fl_filename, alpha_temp_path, dataset_id, "fluorescent")

            # Save to DB via Next.js admin API
            await post_status(session, upload_id, user_id, {
                "status": "processing",
                "progress": 80,
                "message": "Saving dataset..."
            })

            admin_payload = {
                "action": "dataset",
                "datasetId": dataset_id,
                "name": name,
                "description": description,
                "institutionId": institution_id,
                "brightfieldBlobUrl": brightfield_blob_url,
                "fluorescentBlobUrl": fluorescent_blob_url,
                "spacing": float(spacing) if spacing else None,
                "brightfieldNumZ": bfZ,
                "brightfieldNumY": bfY,
                "brightfieldNumX": bfX,
                "fluorescentNumZ": flZ,
                "fluorescentNumY": flY,
                "fluorescentNumX": flX,
            }

            async with session.post(f"{NEXT_BASE_URL}/api/admin", json=admin_payload) as resp:
                admin_result = await resp.json()
                if resp.status >= 300:
                    await post_status(session, upload_id, user_id, {
                        "status": "failed",
                        "progress": 0,
                        "message": "Failed to save dataset",
                        "error": admin_result.get("error") if isinstance(admin_result, dict) else str(admin_result)
                    })
                    return

            await post_status(session, upload_id, user_id, {
                "status": "completed",
                "progress": 100,
                "message": "Dataset uploaded successfully!",
                "result": {"datasetId": dataset_id}
            })

        except Exception as e:
            logger.exception("Processing failed: %s", e)
            await post_status(session, upload_id, user_id, {
                "status": "failed",
                "progress": 0,
                "message": "Cancelled" if str(e).lower().startswith("cancel") else "Processing failed",
                "error": str(e)
            })
        finally:
            if alpha_temp_path and os.path.exists(alpha_temp_path):
                os.remove(alpha_temp_path)


@app.post("/process-dataset")
async def process_dataset(
    name: str = Form(...),
    description: str = Form(default=""),
    institutionId: str = Form(...),
    brightfieldTempUrl: Optional[str] = Form(default=None),
    fluorescentTempUrl: Optional[str] = Form(default=None),
    alphaTempUrl: Optional[str] = Form(default=None),
    brightfieldFilename: Optional[str] = Form(default=None),
    fluorescentFilename: Optional[str] = Form(default=None),
    alphaFilename: Optional[str] = Form(default=None),
    spacing: str = Form(default=""),
    uploadId: str = Form(...),
    userId: str = Form(...),
    nextBaseUrl: str = Form(default=None)
):
    global NEXT_BASE_URL
    if nextBaseUrl:
        NEXT_BASE_URL = nextBaseUrl

    # Schedule background coroutine on current event loop and return immediately
    asyncio.create_task(
        process_dataset_bg(uploadId, userId, name, description, institutionId, spacing,
                           brightfieldTempUrl, brightfieldFilename,
                           fluorescentTempUrl, fluorescentFilename,
                           alphaTempUrl)
    )

    return {"accepted": True, "uploadId": uploadId}


from fastapi import Request, Header

@app.post("/cancel")
async def cancel_upload(request: Request, x_internal_secret: Optional[str] = Header(default=None)):
    if not INTERNAL_API_SECRET or x_internal_secret != INTERNAL_API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    upload_id = body.get("uploadId")
    if not upload_id:
        raise HTTPException(status_code=400, detail="uploadId is required")
    # Mark cancel flag; process loop periodically checks this
    CANCEL_FLAGS[upload_id] = True
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
