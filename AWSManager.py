import s3fs
import boto3

s3 = boto3.resource('s3')
bucket_name_features='flashcrash_features'


# Call S3 to list current buckets
response = s3.list_buckets()
# Get a list of all bucket names from the response
buckets = [bucket['Name'] for bucket in response['Buckets']]
# Print out the bucket list
print("Bucket List: %s" % buckets)


if bucket_name_features not in buckets:
    s3.create_bucket(Bucket=bucket_name_features)

#s3.Object(bucket_name_features, 'hello.txt').put(Body=open('/tmp/hello.txt', 'rb'))



ec2 = boto3.client('ec2')
response = ec2.describe_instances()
print(response)