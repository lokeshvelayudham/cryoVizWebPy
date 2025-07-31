from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from skimage import io
from PIL import Image
import numpy as np
from azure.storage.blob import BlobServiceClient
from typing import Optional
import logging
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Blob Storage configuration
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in environment")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
storage_account_name = "bivlargefiles"  # Explicitly set the storage account name
container_name = "cryovizweb"  # Container within the storage account

async def process_tiff_stack(tiff_file: Optional[UploadFile], alpha_file: Optional[UploadFile], dataset_id: str, modality: str) -> Optional[str]:
    if not tiff_file:
        return None

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_tiff_path = os.path.join(temp_dir, tiff_file.filename)
    temp_alpha_path = None
    try:
        with open(temp_tiff_path, "wb") as f:
            f.write(await tiff_file.read())

        logger.debug(f"Reading TIFF: {temp_tiff_path}")
        rgba_stack = io.imread(temp_tiff_path)
        if rgba_stack.ndim != 4 or rgba_stack.shape[-1] != 4:
            raise HTTPException(status_code=400, detail=f"Invalid TIFF format for {modality}: must be RGBA with shape (Z, Y, X, 4)")

        Z, Y, X, _ = rgba_stack.shape
        alpha_stack = None
        if alpha_file:
            temp_alpha_path = os.path.join(temp_dir, alpha_file.filename)
            with open(temp_alpha_path, "wb") as f:
                f.write(await alpha_file.read())
            logger.debug(f"Reading Alpha: {temp_alpha_path}")
            alpha_stack = io.imread(temp_alpha_path)
            if alpha_stack.shape != (Z, Y, X):
                raise HTTPException(status_code=400, detail=f"Alpha mask dimensions {alpha_stack.shape} do not match {modality} stack {rgba_stack.shape}")

        alpha_threshold = 10

        def apply_alpha_mask(rgba, alpha_mask=None):
            rgba = rgba.copy()
            if alpha_mask is not None:
                mask = alpha_mask < alpha_threshold
                rgba[mask] = [0, 0, 0, 0]
            return rgba

        # Use the correct storage account and container
        container_client = blob_service_client.get_container_client(container_name)
        blob_prefix = f"dataset-{dataset_id}/{modality}"

        # Save XY slices
        for z in range(Z):
            rgba_slice = rgba_stack[z]
            alpha_mask = alpha_stack[z] if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_slice, alpha_mask))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xy/{z:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            logger.debug(f"Uploading to Azure: {blob_name}")
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save XZ slices
        for y in range(Y):
            rgba_xz = np.stack([rgba_stack[z, y, :, :] for z in range(Z)], axis=0)
            alpha_xz = np.stack([alpha_stack[z, y, :] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_xz, alpha_xz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/xz/{y:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            logger.debug(f"Uploading to Azure: {blob_name}")
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Save YZ slices
        for x in range(X):
            rgba_yz = np.stack([rgba_stack[z, :, x, :] for z in range(Z)], axis=0)
            alpha_yz = np.stack([alpha_stack[z, :, x] for z in range(Z)], axis=0) if alpha_stack is not None else None
            img = Image.fromarray(apply_alpha_mask(rgba_yz, alpha_yz))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            blob_name = f"{blob_prefix}/yz/{x:03d}.png"
            blob_client = container_client.get_blob_client(blob_name)
            logger.debug(f"Uploading to Azure: {blob_name}")
            blob_client.upload_blob(img_byte_arr.read(), overwrite=True)

        # Count images in each folder
        num_z = len(list(container_client.list_blobs(f"{blob_prefix}/xy")))-1
        print(f"num_z: {num_z}")
        num_y = len(list(container_client.list_blobs(f"{blob_prefix}/xz")))-1
        print(f"num_y: {num_y}")
        num_x = len(list(container_client.list_blobs(f"{blob_prefix}/yz")))-1
        print(f"num_x: {num_x}")

        
        

        # Construct the blob URL with the correct storage account
        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_prefix}"
        logger.info(f"Successfully processed and uploaded {modality} for dataset {dataset_id}")
        logger.info(f"num_z: {num_z}")
        logger.info(f"num_y: {num_y}")
        logger.info(f"num_x: {num_x}")
        return blob_url, num_z, num_y, num_x
    except Exception as e:
        logger.error(f"Error processing {modality} TIFF: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)
        if temp_alpha_path and os.path.exists(temp_alpha_path):
            os.remove(temp_alpha_path)

async def upload_tiff(tiff_file: Optional[UploadFile], dataset_id: str, filename: str) -> Optional[str]:
    if not tiff_file:
        return None
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(f"dataset-{dataset_id}/{filename}")
        logger.debug(f"Uploading to Azure: dataset-{dataset_id}/{filename}")
        blob_client.upload_blob(await tiff_file.read(), overwrite=True)
        return f"https://{storage_account_name}.blob.core.windows.net/{container_name}/dataset-{dataset_id}/{filename}"
    except Exception as e:
        logger.error(f"Error uploading {filename}: {str(e)}")
        raise

@app.post("/process-dataset")
async def process_dataset(
    name: str = Form(...),
    description: str = Form(default=""),
    institutionId: str = Form(...),
    brightfield: Optional[UploadFile] = File(None),
    fluorescent: Optional[UploadFile] = File(None),
    alpha: Optional[UploadFile] = File(None),
    liverTiff: Optional[UploadFile] = File(None),
    tumorTiff: Optional[UploadFile] = File(None),
    voxels: str = Form(default=""),
    thickness: str = Form(default=""),
    pixelLengthUM: str = Form(default=""),
    zSkip: str = Form(default=""),
    specimen: str = Form(default=""),
    pi: str = Form(default=""),
    dims3X: str = Form(default=""),
    dims3Y: str = Form(default=""),
    dims3Z: str = Form(default=""),
    dims2X: str = Form(default=""),
    dims2Y: str = Form(default=""),
    dims2Z: str = Form(default=""),
    imageDimsX: str = Form(default=""),
    imageDimsY: str = Form(default=""),
    imageDimsZ: str = Form(default=""),
    assignedUsers: str = Form(default="[]"),
):
    dataset_id = str(uuid.uuid4())
    try:
        logger.info(f"Processing dataset {dataset_id}")
        brightfield_result = await process_tiff_stack(brightfield, alpha, dataset_id, "brightfield") if brightfield else (None, None, None, None)
        fluorescent_result = await process_tiff_stack(fluorescent, alpha, dataset_id, "fluorescent") if fluorescent else (None, None, None, None)
        
        brightfield_blob_url, brightfield_num_z, brightfield_num_y, brightfield_num_x = brightfield_result
        fluorescent_blob_url, fluorescent_num_z, fluorescent_num_y, fluorescent_num_x = fluorescent_result

        liverTiffBlobUrl = await upload_tiff(liverTiff, dataset_id, "liverTiff") if liverTiff else None
        tumorTiffBlobUrl = await upload_tiff(tumorTiff, dataset_id, "tumorTiff") if tumorTiff else None

        alphaBlobUrl = None  # Alpha is processed with brightfield/fluorescent, no separate blob

        return {
            "datasetId": dataset_id,
            "name": name,
            "description": description,
            "institutionId": institutionId,
            "brightfieldBlobUrl": brightfield_blob_url,
            "fluorescentBlobUrl": fluorescent_blob_url,
            "alphaBlobUrl": alphaBlobUrl,
            "liverTiffBlobUrl": liverTiffBlobUrl,
            "tumorTiffBlobUrl": tumorTiffBlobUrl,
            "voxels": float(voxels) if voxels else None,
            "thickness": float(thickness) if thickness else None,
            "pixelLengthUM": float(pixelLengthUM) if pixelLengthUM else None,
            "zSkip": int(zSkip) if zSkip else None,
            "specimen": specimen,
            "pi": pi,
            "dims3X": int(dims3X) if dims3X else None,
            "dims3Y": int(dims3Y) if dims3Y else None,
            "dims3Z": int(dims3Z) if dims3Z else None,
            "dims2X": int(dims2X) if dims2X else None,
            "dims2Y": int(dims2Y) if dims2Y else None,
            "dims2Z": int(dims2Z) if dims2Z else None,
            "imageDimsX": int(imageDimsX) if imageDimsX else None,
            "imageDimsY": int(imageDimsY) if imageDimsY else None,
            "imageDimsZ": int(imageDimsZ) if imageDimsZ else None,
            "assignedUsers": eval(assignedUsers),  # Parse JSON string to list
            "brightfieldNumZ": brightfield_num_z,
            "brightfieldNumY": brightfield_num_y,
            "brightfieldNumX": brightfield_num_x,
            "fluorescentNumZ": fluorescent_num_z,
            "fluorescentNumY": fluorescent_num_y,
            "fluorescentNumX": fluorescent_num_x,
        }
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)