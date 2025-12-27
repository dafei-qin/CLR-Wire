import os
import io
import time
from subprocess import PIPE, Popen
from boto3.session import Session, Config
import boto3
from botocore.exceptions import ClientError
from utils.common import init_logger

def get_file(s3, url: str):
    if url.startswith("s3://"):
        # new fetch (cossign)
        return s3.download_file_bytes(url, retry_times=3, log_detail=False)
        # return s3.download_file_bytes(url, log_detail=False)
    else:
        return open(url, "rb")

def get_image(s3, url: str):
    if url.startswith("s3://"):
        return s3.download_file_bytes(url, log_detail=False)
    else:
        return url


class S3(object):
    def __init__(self, aws_access_key_id, aws_secret_access_key, endpoint_url, config=None, region_name=None,):
        self.session = Session(aws_access_key_id=aws_access_key_id,
                               aws_secret_access_key=aws_secret_access_key)
        self.s3 = self.session.client("s3", endpoint_url=endpoint_url, config=config, region_name=region_name,)

    def is_file(self, s3_path):
        """
        判断文件是否存在
        :param s3_path:
        :return:
        """
        bucket, path = self.parse_path(s3_path)
        try:
            self.s3.head_object(Bucket=bucket, Key=path)
        except ClientError as e:
            if int(e.response['Error']['Code']) == 404:
                return False
            else:
                raise ValueError(f"An error occurred ({e.response['Error']['Code']}) "
                                 f"when calling the s3_is_file, {e.response['Error']['Message']}")
        return True

    def is_dir(self, s3_path):
        """
        判断文件夹是否存在，这里会检查有没有加/，没有会帮补上
        :param s3_path:
        :return:
        """
        bucket, path = self.parse_path(s3_path)
        if not path.endswith('/') and path:
            path += '/'
        res = self.s3.list_objects(Bucket=bucket, Prefix=path)
        return 'Contents' in res

    def last_modify_time(self, s3_path):
        """
        查询文件的最新的修改时间
        :param s3_path:
        :return:
        """
        bucket, path = self.parse_path(s3_path)
        res = self.s3.list_objects(Bucket=bucket, Prefix=path)
        if 'Contents' in res:
            for r in res['Contents']:
                if r['Key'] == path:
                    return r['LastModified']
        raise ValueError(f"{s3_path}  Not Found")

    def exists(self, s3_path):
        """
        判断文件是否存在，基本没有用到
        :param s3_path:
        :return:
        """
        return self.is_file(s3_path) or self.is_dir(s3_path)

    def list_dir(self, s3_path):
        """
        查看一个目录下的文件的路径
        :param s3_path:
        :return:
        """
        bucket, path = self.parse_path(s3_path)
        if not path.endswith('/') and path:
            path += '/'
        file_key = []
        marker = ''
        while True:
            res = self.s3.list_objects(Bucket=bucket, Prefix=path, Marker=marker)
            if 'Contents' not in res:
                raise ValueError(f"{s3_path}  fold not exist")
            for key in res['Contents']:
                file_key.append(key['Key'])

            if not res['IsTruncated']:
                break
            else:
                marker = res['NextMarker']

        return file_key

    def list_dir_detail(self, s3_path):
        """
        查看一个目录下的文件的路径，并且返回对应的文件详情
        :param s3_path:
        :return:
        """
        bucket, path = self.parse_path(s3_path)
        if not path.endswith('/') and path:
            path += '/'
        res = self.s3.list_objects(Bucket=bucket, Prefix=path)
        if 'Contents' not in res:
            raise ValueError(f"{s3_path}  fold not exist")
        file_details = []
        len_dir = len(path.split('/'))
        for content in res['Contents']:
            if content['Size'] == 0:
                is_dir = True
            else:
                is_dir = False
            file_detail = {
                "file_name": content['Key'],
                "modify_time": content['LastModified'].strftime("%Y-%m-%d %H:%M"),
                "file_size": content['Size'],
                "is_dir": is_dir
            }
            file_name = content['Key']
            if file_name.endswith('/'):
                file_name = file_name[0:-1]
            if len(file_name.split('/')) == len_dir:
                file_details.append(file_detail)
        return file_details

    def _upload(self, bucket, key_path, path):
        with open(path, "rb") as f:
            self.s3.put_object(Bucket=bucket, Key=key_path, Body=f.read(), ACL='public-read-write')

    def _chunk_upload(self, bucket, key_path, path, chunk_size=1024 * 1024 * 10):
        # get file size
        filesize = os.path.getsize(path)
        if filesize <= chunk_size:
            self._upload(bucket, key_path, path)
            return

        response = self.s3.create_multipart_upload(Bucket=bucket, Key=key_path)
        upload_id = response['UploadId']

        with open(path, 'rb') as f:
            parts = []
            part_number = 1
            while True:
                chunk_contents = f.read(chunk_size)
                if not chunk_contents:
                    break

                response = self.s3.upload_part(
                    Bucket=bucket,
                    Key=key_path,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk_contents
                )

                parts.append({
                    'PartNumber': part_number,
                    'ETag': response['ETag']
                })

                part_number += 1

        self.s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key_path,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )

    def upload_file(self, path, s3_path, chunk_size=None):
        """
        上传文件的API，从本地上传到s3上，如果输入的s3为目录，取上传文件路径的文件名为s3文件上的文件名
        :param path: 本地路径
        :param s3_path: s3要存储的路径
        :return:
        """
        bucket, key_path = self.parse_path(s3_path)
        if s3_path.endswith('/'):
            key_path += os.path.basename(path)
        if chunk_size:
            self._chunk_upload(bucket, key_path, path, chunk_size)
        else:
            self._upload(bucket, key_path, path)
        return

    def upload_files(self, path, s3_path, chunk_size=None):
        """
        从本地上传一个目录到s3上，实际上还是得一个文件一个文件的去传，如果有需要改为多进程去传，目前需求量较小
        :param path:
        :param s3_path:
        :return:
        """
        bucket, key_path_prefix = self.parse_path(s3_path)
        if key_path_prefix.endswith('/'):
            key_path_prefix = key_path_prefix[:-1]
        if path.endswith('/'):
            path = path[:-1]
        for root, dirs, files in os.walk(path):
            for file in files:
                suffix_path = os.path.join(root, file)[len(path) + 1:]
                key_path = os.path.join(key_path_prefix, suffix_path)
                path = os.path.join(root, file)
                if chunk_size:
                    self._chunk_upload(bucket, key_path, path, chunk_size)
                else:
                    self._upload(bucket, key_path, path)
        return

    def upload_fileobj(self, file, path_key):
        """
        上传一个文件对象，以fileobj格式上传
        :param file:
        :param path_key:
        :return:
        """
        bucket, key = self.parse_path(path_key)
        self.s3.put_object(Bucket=bucket, Key=key, Body=file, ACL='public-read-write')

    def _download(self, bucket, key_path, path):
        """
        执行下载的函数，files和file都需要使用
        :param bucket:
        :param key_path:
        :param path:
        :return:
        """
        res = self.s3.get_object(Bucket=bucket, Key=key_path)
        with open(path, 'wb') as f:
            f.write(res['Body'].read())
        return

    def _chunk_download(self, bucket, key_path, path, chunk_size=1024 * 1024 * 10):
        """
        执行下载的函数，files和file都需要使用
        :param bucket:
        :param key_path:
        :param path:
        :param chunk_size:
        :return:
        """
        response = self.s3.head_object(Bucket=bucket, Key=key_path)
        file_size = int(response['ContentLength'])
        if file_size <= chunk_size:
            self._download(bucket, key_path, path)
            return

        num_chunks = file_size // chunk_size + 1

        with open(path, 'wb') as f:
            for i in range(num_chunks):
                range_start = i * chunk_size
                range_end = min((i + 1) * chunk_size - 1, file_size - 1)

                response = self.s3.get_object(
                    Bucket=bucket,
                    Key=key_path,
                    Range=f'bytes={range_start}-{range_end}'
                )

                chunk_contents = response['Body'].read()
                f.write(chunk_contents)

    def download_file(self, s3_path, local_path, chunk_size=None):
        """
        从s3上下载一个文件到本地
        :param s3_path:
        :param local_path:
        :return:
        """
        bucket, key_path = self.parse_path(s3_path)
        if not self.is_file(s3_path):
            raise ValueError(f"file {s3_path} is not exist")
        if key_path.endswith("/"):
            raise ValueError(f"{s3_path} is not a file")
        if local_path.endswith("/"):
            local_path = os.path.join(local_path, os.path.basename(key_path))
        dir_name = os.path.dirname(local_path)
        if not local_path.startswith("/"):
            dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if chunk_size:
            self._chunk_download(bucket, key_path, local_path, chunk_size)
        else:
            self._download(bucket, key_path, local_path)
        return

    def download_files(self, s3_path, local_path=None, chunk_size=None):
        """
        从s3上下载一个目录下的文件到本地
        :param s3_path:
        :param local_path:
        :return:
        """
        bucket, key_path = self.parse_path(s3_path)
        if not key_path.endswith("/"):
            key_path += "/"
        res_list = self.s3.list_objects(Bucket=bucket, Prefix=key_path)
        if 'Contents' not in res_list:
            raise ValueError(f"{s3_path} s3 file note exist")
        else:
            for key in res_list['Contents']:
                file_key = key['Key']
                if local_path:
                    if not local_path.startswith("/"):
                        dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), local_path)
                    else:
                        dir_name = local_path
                    file_path = os.path.join(dir_name, file_key[len(key_path):])
                else:
                    dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            os.path.split(key_path[:-1])[-1])
                    file_path = os.path.join(dir_name, file_key[len(key_path) + 1:])
                local_file_path = os.path.join(local_path, file_path)
                dir_name_mk = os.path.dirname(local_file_path)
                if not os.path.exists(dir_name_mk):
                    os.makedirs(dir_name_mk)
                if file_path.endswith("/"):
                    continue
                if chunk_size:
                    self._chunk_download(bucket, file_key, local_file_path, chunk_size)
                else:
                    self._download(bucket, file_key, local_file_path)
        return

    def download_fileobj(self, path_key, filename):
        """
        下载一个文件对象，这个是从远端上传
        :param file:
        :param path_key:
        :return:
        """
        bucket, key = self.parse_path(path_key)

        with open(filename, 'wb') as data:
            self.s3.download_fileobj(bucket, key, data)
        return data

    def download(self, s3_path, local_path=None, chunk_size=None):
        """
        在不知道路径是文件还是目录时，需要发请求知道是什么类型，文件or文件夹。
        优先检查是否存在文件，否者检查是否存在文件夹。在知道下载是文件或文件夹时尽量不用
        :param s3_path:
        :param local_path:
        :return:
        """
        if self.is_file(s3_path):
            if not local_path:
                _, key_path = self.parse_path(s3_path)
                self.download_file(s3_path, os.path.basename(key_path), chunk_size)
            else:
                self.download_file(s3_path, local_path, chunk_size)
        elif self.is_dir(s3_path):
            self.download_files(s3_path, local_path, chunk_size)
        else:
            raise ValueError(f"{s3_path} is not exist")

    def delete_file(self, s3_path):
        """
        从s3上删除掉文件
        :param s3_path:
        :return:
        """
        if self.is_file(s3_path):
            bucket, key_path = self.parse_path(s3_path)
            self.s3.delete_object(Bucket=bucket, Key=key_path)
        else:
            raise ValueError(f"{s3_path} not found")

    def delete_files(self, s3_path):
        """
        给出s3的目录名称，会根据目录名称取删除文件夹的名称，用list把要删除的文件名称保存，一次请求完成整个删除过程
        :param s3_path:
        :return:
        """
        if self.is_dir(s3_path):
            bucket, key_path = self.parse_path(s3_path)
            res = self.list_dir(s3_path)
            objs = []
            for key in res:
                obj = {'Key': key}
                objs.append(obj)
            del_obj = {'Objects': objs, 'Quiet': True}
            self.s3.delete_objects(Bucket=bucket, Delete=del_obj)
            return
        else:
            raise ValueError(f"{s3_path} is not exist")

    def delete(self, s3_path):
        """
        不知道要删除的是文件还是文件夹时使用，需要发请求知道是什么类型，知道类型尽量不用
        :param s3_path:
        :return:
        """
        if self.is_file(s3_path):
            self.delete_file(s3_path)
        elif self.is_dir(s3_path):
            self.delete_files(s3_path)
        else:
            raise ValueError(f"{s3_path}  is not exist")

    def bucket_copy_file(self, s3_path_src, s3_path_des):
        """
        拷贝s3文件，必须是一个集群内的文件，不在一个集群内的文件会检查不到，s3客户端选择为目标桶的客户端
        :param s3_path_src:
        :param s3_path_des:
        :return:
        """
        if not self.is_file(s3_path_src):
            raise ValueError(f"{s3_path_src} is not a file")
        bucket_src, src_key_path = self.parse_path(s3_path_src)
        bucket_des, des_key_path = self.parse_path(s3_path_des)
        self.s3.copy_object(Bucket=bucket_des, Key=des_key_path, CopySource=str(bucket_src + '/' + src_key_path),
                            ACL='public-read-write')

    def bucket_copy_files(self, s3_dir_path_src, s3_dir_path_des):
        """
        拷贝文件夹，原始目录到目标目录，实际上也是一个文件一个文件的拷贝
        :param s3_dir_path_src:
        :param s3_dir_path_des:
        :return:
        """
        if s3_dir_path_src.endswith("/"):
            s3_dir_path_src = s3_dir_path_src[:-1]
        if not self.is_dir(s3_dir_path_src):
            raise ValueError(f"{s3_dir_path_src} is not exist")
        files_in_dir = self.list_dir(s3_dir_path_src)
        bucket_src, dir_path_src = self.parse_path(s3_dir_path_src)
        bucket_des, dir_path_des = self.parse_path(s3_dir_path_des)
        if dir_path_des.endswith("/"):
            dir_path_des = dir_path_des[:-1]

        for src_key in files_in_dir:
            des_key = os.path.join(dir_path_des, src_key[len(dir_path_src) + 1:])
            self.s3.copy_object(Bucket=bucket_des, Key=des_key, CopySource=str(bucket_src + '/' + src_key),
                                ACL='public-read-write')

    def bucket_copy(self, s3_dir_path_src, s3_dir_path_des):
        """
        不知道是不是文件，可能是文件夹，先用请求去知道是文件还是文件夹，然后再拷贝，优先文件，确定没有文件之后才能去确
        定文件夹
        :param s3_dir_path_src:
        :param s3_dir_path_des:
        :return:
        """
        if not s3_dir_path_src.endswith("/") and not s3_dir_path_des.endswith("/"):
            self.bucket_copy_file(s3_dir_path_src, s3_dir_path_des)
        elif s3_dir_path_src.endswith("/") and s3_dir_path_des.endswith("/"):
            self.bucket_copy_files(s3_dir_path_src, s3_dir_path_des)
        else:
            raise ValueError(f"{s3_dir_path_src} or {s3_dir_path_src} must all be file or dir")

    def open_file(self, s3_path):
        """
        以字节流的形式去访问文件，而不需要要下载到本地
        :param s3_path:
        :return:
        """
        bucket, key_path = self.parse_path(s3_path)
        if not self.is_file(s3_path):
            raise ValueError(f"{s3_path} is not exist")
        if key_path.endswith("/"):
            raise ValueError(f"{s3_path}  is not  a file")
        res = self.s3.get_object(Bucket=bucket, Key=key_path)
        return res['Body']._raw_stream

    def write_file(self, byte_stream, s3_path):
        """
        将文件以字节流的形式写到目标的文件中
        :param byte_stream:
        :param s3_path:
        :return:
        """
        bucket, key_path = self.parse_path(s3_path)
        self.s3.put_object(Bucket=bucket, Key=key_path, Body=byte_stream, ACL='public-read-write')
        return

    @staticmethod
    def parse_path(s3_path=None):
        """
        从一个s3规定的路径中找到桶的名字和对应的要存储的文件名
        :param s3_path:
        :return:
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(f"{s3_path} must start with s3://")
        s3_valid_path = s3_path[5:]
        bucket, file_path = s3_valid_path.split("/", 1)
        return bucket, file_path

    @classmethod
    def transfer_data(cls, src_access_key_id, src_secret_access_key, src_endpoint_url,
                      dst_access_key_id, dst_secret_access_key, dst_endpoint_url,
                      src_path, dst_path):
        """
        从两个不同的集群间transfer数据。path可以为dir、或file。
        """
        src_client = cls(src_access_key_id, src_secret_access_key, src_endpoint_url)
        dst_client = cls(dst_access_key_id, dst_secret_access_key, dst_endpoint_url)

        src_resource = src_client.session.resource('s3', endpoint_url=src_endpoint_url)
        dst_resource = dst_client.session.resource('s3', endpoint_url=dst_endpoint_url)

        def transfer_one_file(src_bucket, src_file_path, dst_bucket, dst_file_path):
            try:
                src_obj = src_resource.Object(src_bucket, src_file_path)
                dst_obj = dst_resource.Object(dst_bucket, dst_file_path)
                # https://iwiki.woa.com/pages/viewpage.action?pageId=112001577，第6点
                # 如果不加这个ACL，可能出现权限问题
                dst_obj.put(ACL="public-read-write",
                            Body=src_obj.get()['Body'].read())
            except Exception as e:
                raise RuntimeError(
                    f"{e}. "
                    f"{src_bucket}:{src_file_path} --> {dst_bucket}:{dst_file_path}")

        def transfer_dir(src_dir, dst_dir):
            if src_dir.endswith("/"):
                src_dir = src_dir[:-1]
            if not src_client.is_dir(src_dir):
                raise ValueError(f"{src_dir} is not exist")
            files_in_dir = src_client.list_dir(src_dir)
            bucket_src, dir_path_src = cls.parse_path(src_dir)
            bucket_dst, dir_path_dst = cls.parse_path(dst_dir)
            if dir_path_dst.endswith("/"):
                dir_path_dst = dir_path_dst[:-1]

            for src_key in files_in_dir:
                dst_key = os.path.join(dir_path_dst, src_key[len(dir_path_src) + 1:])
                transfer_one_file(bucket_src, src_key, bucket_dst, dst_key)

        if not src_path.endswith("/") and not dst_path.endswith("/"):
            transfer_one_file(*cls.parse_path(src_path), *cls.parse_path(dst_path))
        elif src_path.endswith("/") and dst_path.endswith("/"):
            transfer_dir(src_path, dst_path)
        else:
            raise ValueError(f"{src_path} or {dst_path} must all be file or dir")



class AwsS3CosHelper:
    def __init__(self):
        cos_config = Config(signature_version='v4', s3={'addressing_style': 'virtual'})
        
        endpoint_url = 'https://s3.us-east-2.amazonaws.com'
        region = 'us-east-2'
        self.cos_client = S3(aws_access_key_id=os.environ.get('COS_ACCESS_KEY'),
                             aws_secret_access_key=os.environ.get('COS_SECRET_KEY'),
                             endpoint_url=endpoint_url,
                             config=cos_config,
                             region_name=region,)

        self.cos_buckets = [f'artrefine-asset-{idx}-1258344700' for idx in range(8)]
        self.logger = init_logger()

    def convert_url(self, s3_path):
        bucket, file_path = self.cos_client.parse_path(s3_path)
        hash_int = int(file_path.split("/", 1)[0])
        bucket = self.cos_buckets[hash_int % len(self.cos_buckets)]

        cos_path = f"s3://{bucket}/{file_path}"
        return cos_path

    def upload_file(self, path, s3_path, retry_times=3, direct_path=True):
        cos_path = s3_path if direct_path else self.convert_url(s3_path)
        # self.logger.info(f"upload {path} to {s3_path}, cos_path is {cos_path}")

        bucket, file_path = self.cos_client.parse_path(cos_path)

        for ii in range(retry_times):
            try:
                with open(path, 'rb') as f:
                    self.cos_client.s3.put_object(Bucket=bucket, Key=file_path, Body=f)
                return
            except Exception as e:
                self.logger.warning(f"upload {path} to {cos_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

    def upload_file_bytes(self, file_bytes, s3_path, retry_times=3, direct_path=True, log_detail=False):
        cos_path = s3_path if direct_path else self.convert_url(s3_path)
        if log_detail:
            self.logger.info(f"upload file_bytes to {s3_path}, cos_path is {cos_path}")

        bucket, file_path = self.cos_client.parse_path(cos_path)

        for ii in range(retry_times):
            try:
                self.cos_client.s3.put_object(Bucket=bucket, Key=file_path, Body=file_bytes)
                return
            except Exception as e:
                self.logger.warning(f"upload file_bytes to {cos_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

    def download_file(self, s3_path, local_path, retry_times=3, direct_path=False):
        cos_path = s3_path
        
        bucket, file_path = self.cos_client.parse_path(cos_path)

        for ii in range(retry_times):
            try:
                obj = self.cos_client.s3.get_object(Bucket=bucket, Key=file_path)
                with open(local_path, "wb") as f:
                    for chunk in obj['Body'].iter_chunks(1024 * 1024 * 16):
                        f.write(chunk)
                return
            except Exception as e:
                self.logger.warning(f"download {cos_path} to {local_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

    def download_file_bytes(self, s3_path, retry_times=3, log_detail=True, direct_path=True):
        cos_path = s3_path
        if log_detail:
            self.logger.info(f"download bytes from {s3_path}, cos_path is {cos_path}")

        bucket, file_path = self.cos_client.parse_path(cos_path)

        for ii in range(retry_times):
            try:
                obj = self.cos_client.s3.get_object(Bucket=bucket, Key=file_path)
                data = obj['Body'].read()
                data_stream = io.BytesIO(data)
                return data_stream
            except Exception as e:
                self.logger.warning(f"download {cos_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

class S3cmdHelper:
    def __init__(self):
        self.cos_buckets = [f'artrefine-asset-{idx}-1258344700' for idx in range(8)]
        self.logger = init_logger()

    def convert_url(self, s3_path):
        bucket, file_path = S3.parse_path(s3_path)
        hash_int = int(file_path.split("/", 1)[0])
        bucket = self.cos_buckets[hash_int % len(self.cos_buckets)]

        cos_path = f"s3://{bucket}/{file_path}"
        return cos_path

    def upload_file(self, path, s3_path, retry_times=3):
        cos_path = self.convert_url(s3_path)
        self.logger.info(f"upload {path} to {s3_path}, cos_path is {cos_path}")
        proc_args = ['s3cmd', 'put', path, cos_path]

        for ii in range(retry_times):
            try:
                proc = Popen(proc_args)
                status = proc.wait()
                if status == 0:
                    return
                else:
                    raise IOError(f"{proc_args}: exit {status}")
            except Exception as e:
                self.logger.warning(f"upload {path} to {cos_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

    def download_file(self, s3_path, local_path, retry_times=3):
        cos_path = self.convert_url(s3_path)
        self.logger.info(f"download {s3_path} to {local_path}, cos_path is {cos_path}")
        proc_args = ['s3cmd', 'get', cos_path, local_path]

        for ii in range(retry_times):
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)

                proc = Popen(proc_args)
                status = proc.wait()
                if status == 0:
                    return
                else:
                    raise IOError(f"{proc_args}: exit {status}")
            except Exception as e:
                self.logger.warning(f"download {cos_path} to {local_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e

    def download_file_bytes(self, s3_path, retry_times=3, log_detail=True):
        cos_path = self.convert_url(s3_path)
        if log_detail:
            self.logger.info(f"download bytes from {s3_path}, cos_path is {cos_path}")

        proc_args = ['s3cmd', 'get', cos_path, '-']

        for ii in range(retry_times):
            try:
                proc = Popen(proc_args, stdout=PIPE)
                stream = proc.stdout
                result = stream.read()
                status = proc.wait()
                stream.close()
                if status == 0:
                    data_stream = io.BytesIO(result)
                    return data_stream
                else:
                    raise IOError(f"{proc_args}: exit {status}")
            except Exception as e:
                self.logger.warning(f"download {cos_path} failed, retry {ii + 1} times, err: {e}")
                time.sleep(5)
                if ii == retry_times - 1:
                    raise e


if __name__ == '__main__':
    cos_helper = CosHelper()
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/512/00000.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/512/00000.tar')
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/512/00001.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/512/00001.tar')
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/768/00002.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/768/00002.tar')
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/768/00003.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/768/00003.tar')
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/1024/00002.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/1024/00002.tar')
    cos_helper.upload_file(path='/primexu-cfs/aigc/data/demo_sdxl_latent_data/1024/00003.tar',
                           s3_path='s3://asset_external/0000/primexu/tmp/tar_dataset/mock/1024/00003.tar')