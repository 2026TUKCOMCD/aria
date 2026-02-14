import json
import boto3
import base64
import psycopg2
import os
import uuid
from datetime import datetime

# AWS S3 클라이언트 생성 (Lambda 내부에서는 인증키 없이 Role로 동작)
s3_client = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

# DB 정보
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_PORT = os.environ.get("DB_PORT", "5432")

def make_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT
        )
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def lambda_handler(event, context):
    """
    POST /robots/{id}/map 요청 처리
    Body: { "image_base64": "...", "metadata": { ... } }
    """
    conn = None
    try:
        print("Event Received:", event)
        
        # 1. Body 파싱
        body = json.loads(event.get('body', '{}'))
        robot_id = event.get('pathParameters', {}).get('id', 'unknown_robot')
        
        image_data = body.get('image_base64')
        metadata = body.get('metadata', {})
        map_name = body.get('map_name', f"map_{datetime.now().strftime('%Y%m%d_%H%M')}")

        if not image_data:
            return {'statusCode': 400, 'body': json.dumps("No image data provided")}

        # 2. Base64 디코딩 -> 이미지 파일 변환
        image_binary = base64.b64decode(image_data)
        
        # 3. S3 업로드
        # 파일명: maps/robot_01/random_uuid.png (덮어쓰기 방지)
        file_name = f"maps/{robot_id}/{uuid.uuid4()}.png"
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=file_name,
            Body=image_binary,
            ContentType='image/png'
        )
        
        # S3 URL 생성
        # 서울 리전 기준 URL 형식
        s3_url = f"https://{BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{file_name}"
        print(f"S3 Upload Success: {s3_url}")

        # 4. DB 저장
        conn = make_connection()
        cur = conn.cursor()
        
        sql = """
            INSERT INTO robot_maps 
            (robot_id, map_name, s3_url, resolution, width, height, origin_x, origin_y, origin_theta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING map_id;
        """
        
        cur.execute(sql, (
            robot_id, map_name, s3_url,
            metadata.get('resolution'),
            metadata.get('width'),
            metadata.get('height'),
            metadata.get('origin', [0,0,0])[0], # x
            metadata.get('origin', [0,0,0])[1], # y
            metadata.get('origin', [0,0,0])[2]  # theta
        ))
        
        new_map_id = cur.fetchone()[0]
        conn.commit()
        
        return {
            'statusCode': 201,
            'body': json.dumps({
                "message": "Map uploaded successfully",
                "map_id": new_map_id,
                "s3_url": s3_url
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        if conn: conn.rollback()
        return {'statusCode': 500, 'body': json.dumps(str(e))}
    
    finally:
        if conn: conn.close()
