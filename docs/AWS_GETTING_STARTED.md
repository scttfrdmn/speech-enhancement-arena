# AWS Getting Started: First Run

For researchers who have an institutional HPC cluster but have never used AWS directly. From "I have an AWS account" to "the arena is training in the cloud," in seven steps. Assumes a Mac/Linux laptop.

Cloud compute is a legitimate, modest research expense — the full walkthrough below costs well under $1 in actual usage. Treat it the same way you'd treat any other lab expense: budget for it in your grant proposals, get your PI's account or set up a sub-account under the lab's billing, track spend monthly.

---

## What you'll have at the end

- AWS CLI configured locally
- A spot `g6.xlarge` (L4, 24 GB, ~$0.39/hr) running this repo
- A working arena run with logs and checkpoints
- A terminated instance and a confirmed $0 ongoing spend

Total clock time: ~30 minutes. Total compute spend: < $0.50.

---

## Step 1 — Install the AWS CLI

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

Verify:
```bash
aws --version
# aws-cli/2.x.x ...
```

---

## Step 2 — Create an IAM user and access keys

In the AWS console: **IAM → Users → Create user → "Attach policies directly"**. For this walkthrough, attach `AmazonEC2FullAccess` and `AmazonS3FullAccess`. (You'll tighten this later; broad is fine for first-run.)

After creating the user: **Security credentials → Create access key → Command Line Interface (CLI)**. Save the **Access Key ID** and **Secret Access Key** — the secret is shown only once.

> ⚠️ Never commit these to git, never paste them in chat. If you leak them, immediately rotate via the same screen.

---

## Step 3 — Configure the CLI

```bash
aws configure
# AWS Access Key ID:     <paste>
# AWS Secret Access Key: <paste>
# Default region name:   us-west-2     # or wherever L4 spot is cheap right now
# Default output format: json
```

Verify:
```bash
aws sts get-caller-identity
# {"UserId": "...", "Account": "...", "Arn": "arn:aws:iam::.../<your-user>"}
```

---

## Step 4 — Create a key pair (for SSH)

```bash
aws ec2 create-key-pair --key-name research-laptop \
  --query 'KeyMaterial' --output text > ~/.ssh/research-laptop.pem
chmod 400 ~/.ssh/research-laptop.pem
```

You now have a private key on your laptop that AWS will trust for SSH access.

---

## Step 5 — Open a security group on port 22 and 8765

```bash
# Find your default VPC (already exists in every account)
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)

# Create a security group
aws ec2 create-security-group --group-name arena-demo \
  --description "Speech Arena demo SSH + WebSocket" --vpc-id "$VPC"

SG=$(aws ec2 describe-security-groups --group-names arena-demo \
  --query 'SecurityGroups[0].GroupId' --output text)

# Allow SSH and the streaming demo port from your current IP only
MYIP=$(curl -s ifconfig.me)/32
aws ec2 authorize-security-group-ingress --group-id "$SG" \
  --protocol tcp --port 22 --cidr "$MYIP"
aws ec2 authorize-security-group-ingress --group-id "$SG" \
  --protocol tcp --port 8765 --cidr "$MYIP"
```

---

## Step 6 — Launch a g6.xlarge spot instance

```bash
# Find the latest Deep Learning AMI (Ubuntu 22.04, PyTorch GPU)
AMI=$(aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*' \
            'Name=state,Values=available' \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
echo "AMI: $AMI"

# Launch one g6.xlarge as a spot instance
aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type g6.xlarge \
  --key-name research-laptop \
  --security-group-ids "$SG" \
  --instance-market-options 'MarketType=spot' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=arena-demo}]' \
  --query 'Instances[0].InstanceId' --output text
# i-xxxxxxxxxxxxxxxxx
```

Get the public IP once it's running (takes ~30 seconds):
```bash
aws ec2 describe-instances --filters Name=tag:Name,Values=arena-demo \
                                     Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text
```

**What just happened:** you placed a spot bid at the current market price (probably ~$0.30/hr right now). The instance has the NVIDIA driver, CUDA, and PyTorch preinstalled.

---

## Step 7 — SSH in and run the arena

```bash
ssh -i ~/.ssh/research-laptop.pem ubuntu@<public-ip>

# On the instance:
git clone https://github.com/scttfrdmn/speech-enhancement-arena.git
cd speech-enhancement-arena
pip install -r requirements.txt

# Smoke test (CPU path works on the GPU instance too — verifies install)
python arena.py --device cpu --epochs 1 --num-samples 64 --batch-size 8

# Real run on the L4
python arena.py --device cuda --epochs 10 --num-samples 1000

# Start the streaming demo
python stream/server/inference.py --checkpoint-dir checkpoints --device cuda \
                                  --host 0.0.0.0 --port 8765
```

On your laptop, open `http://<public-ip>:8765` — you should see the streaming demo. Click Start, speak into your mic, hear the enhancement.

---

## Step 8 — Clean up (don't skip this)

```bash
# Find the instance ID
ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=arena-demo \
                                          Name=instance-state-name,Values=running \
       --query 'Reservations[].Instances[].InstanceId' --output text)

# Terminate it
aws ec2 terminate-instances --instance-ids "$ID"

# Confirm
aws ec2 describe-instances --instance-ids "$ID" \
  --query 'Reservations[].Instances[].State.Name' --output text
# shutting-down  →  terminated
```

Check the next-day billing in **Billing Dashboard → Cost Explorer**. Your full session should be well under $1.

---

## What can go wrong

- **"InsufficientInstanceCapacity"** on the spot launch. The region/AZ ran out of L4 capacity at the spot price. Try a different region (`us-east-1`, `us-west-2`, `eu-west-1`), or switch to on-demand by removing the `--instance-market-options` flag.
- **SSH "Permission denied (publickey)"**. The key file isn't `chmod 400`, or you used the wrong key name in `--key-name`. Re-check both.
- **Browser can't reach `:8765`**. Your laptop's public IP changed (coffee shop, VPN). Re-run the `authorize-security-group-ingress` command with the new `$MYIP`.
- **Spot instance vanished mid-run.** EC2 reclaimed it. This is normal; see [`SPOT_RESILIENCE.md`](SPOT_RESILIENCE.md) for how to survive.

---

---

## The spore.host shortcut (Steps 5–8 in 3 commands)

Once you've done the native walkthrough once, you don't need to do it again. [**spore.host**](https://spore.host) is an open-source CLI suite (`truffle` for capacity search, `spawn` for launch with TTL-based auto-termination, `spored` as the on-instance lifecycle daemon) that collapses Steps 5–8 to:

```bash
# Install (Homebrew on macOS / Linux)
brew install scttfrdmn/tap/truffle scttfrdmn/tap/spawn

# Find the cheapest g6.xlarge spot price across regions
truffle spot g6.xlarge --sort-by-price --active-only

# Launch with a name + TTL — auto-terminates after 4 hours, no manual cleanup
spawn launch --name arena --instance-type g6.xlarge --ttl 4h \
             --on-complete terminate

# Connect by name (DNS resolves arena.spore.host automatically)
spawn connect arena
```

The instance auto-terminates at TTL, so you can't forget the cleanup step and rack up a surprise bill overnight. `spawn extend arena 2h` if your job is running long, `spawn status arena` to check.

---

## See also

- [`SPOT_RESILIENCE.md`](SPOT_RESILIENCE.md) — designing experiments that survive spot interruption (relevant whether you launch via native AWS or spore.host)
- [`HYBRID_CLOUD_WORKFLOW.md`](HYBRID_CLOUD_WORKFLOW.md) — combining institutional clusters with cloud bursts
- [**spore.host on GitHub**](https://github.com/scttfrdmn/spore-host)
