#!/bin/bash
# usage: ./ec2.sh 'remote command'   |  ./ec2.sh --scp local_files... remote_dir
# Ensures the current public IP is allowed on the SSH security group, then runs ssh/scp.
cd "$(dirname "$0")"
export AWS_DEFAULT_REGION=us-east-1
SG=sg-0995c1249715baa48; IP=$(cat .aws_keys/ip.txt); K=.aws_keys/tfb-key.pem
ME=$(curl -s https://checkip.amazonaws.com)
if ! grep -qx "$ME" .aws_keys/allowed_ips.txt 2>/dev/null; then
  aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $ME/32 >/dev/null 2>&1
  echo "$ME" >> .aws_keys/allowed_ips.txt
fi
if [ "$1" = "--scp" ]; then shift; dest="${@: -1}"; set -- "${@:1:$(($#-1))}"; scp -q -o ConnectTimeout=20 -i $K "$@" ubuntu@$IP:$dest
else ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $K ubuntu@$IP "$@"; fi
