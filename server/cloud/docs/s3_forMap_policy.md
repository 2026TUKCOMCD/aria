# 이 파일은 mapUpload를 위한 s3 생성 시 사용되는 정책을 기술한 .md 파일이다

``` json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::aria-map-storage/*" 
        }
    ]
}
```