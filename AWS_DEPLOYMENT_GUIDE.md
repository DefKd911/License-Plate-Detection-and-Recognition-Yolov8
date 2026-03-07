# AWS Deployment Guide: License Plate Detection API

This guide explains **how AWS fits your project**, **cloud terms in plain language**, **free vs paid**, and **step-by-step deployment** for your License Plate Detection FastAPI app.

**Current state:** Your project has **no AWS code yet**. It runs locally with Docker. This guide shows how to deploy that same Docker image on AWS.

---

## Part 1: Cloud Deployment Terms (In Plain Language)

### Terms You’ll See Everywhere

| Term | Meaning (for this project) |
|------|----------------------------|
| **Cloud** | Someone else’s computers (AWS). You run your app on their servers instead of your laptop. |
| **Deploy** | Put your app on those servers so others can call it over the internet (e.g. `https://your-api.aws.com/predict`). |
| **Docker image** | A packaged version of your app (Python + YOLO + EasyOCR + your code). You already have a `Dockerfile` that builds this. |
| **Container** | A running instance of that image. AWS can start/stop containers for you. |
| **Registry** | A place to store Docker images so AWS can pull them. On AWS that’s **ECR** (Elastic Container Registry). |
| **Compute** | The “machine” that runs your code. AWS offers several types (see below). |
| **Region** | Physical location of AWS data centers (e.g. `us-east-1`). You choose one; latency and price can vary. |

### AWS Services You’ll Use

| Service | What it is | Role in this project |
|--------|-------------|----------------------|
| **ECR** (Elastic Container Registry) | Private Docker image storage | You push your built image here; other AWS services pull from it. |
| **EC2** | Virtual server (like a remote PC) | You can install Docker and run your container on a small instance (e.g. t2.micro). |
| **ECS** (Elastic Container Service) | Service to run containers | You define a “task” (your image + CPU/memory). ECS runs it. |
| **Fargate** | “Serverless” way to run ECS tasks | You don’t manage the server; AWS runs your container and bills by vCPU/memory per second. |
| **App Runner** | Fully managed “run my container from ECR” | Easiest: you point it at ECR, set CPU/memory, and get a URL. No ECS/EC2 setup. |
| **IAM** | Identity and permissions | Users and roles that can push to ECR, run ECS/App Runner, etc. |
| **VPC** | Virtual network | Your app runs inside a network. App Runner / Fargate handle most of this for you. |

### How Your Project Fits

- Your app is a **web API** (FastAPI) that:
  - Accepts an image → runs YOLO + EasyOCR → returns JSON or an annotated image.
- It’s **CPU-heavy** (YOLO + EasyOCR), not a tiny script. So:
  - **Good options:** EC2, ECS Fargate, App Runner (all can run Docker with enough CPU/RAM).
  - **Tricky option:** Lambda (limited memory/time and image size; possible but more work).

This guide focuses on **EC2**, **App Runner**, and **ECS Fargate** so you can run the same Docker image you use locally.

---

## Part 2: Do I Need to Pay? Free Tier Explained

### Do you need to purchase anything?

- **No credit card?** You need one to create an AWS account, but you can stay within **Free Tier** and pay **$0** if you use only free offerings and stay under limits.
- **With Free Tier:** You can run this project for **free** in a limited way (e.g. small EC2 or limited Fargate/App Runner usage). Beyond that, you pay only for what you use.

### Types of “Free” on AWS

1. **12‑Month Free Tier** (for new accounts, first 12 months)
   - **EC2:** 750 hours/month of t2.micro (or t3.micro in some regions) = one instance running 24/7.
   - **ECR:** 500 MB of storage (enough for one image of your app).
   - **Data transfer:** Small amount of outbound traffic free (e.g. 15 GB out).
   - **Fargate:** Often included as “X vCPU-hours and Y GB-hours” per month (check current [AWS Free Tier](https://aws.amazon.com/free/) page).

2. **Always Free**
   - **Lambda:** 1 million requests/month (your app is heavy; Lambda is possible but not the main focus here).
   - **ECR:** 500 MB storage can be always free in some cases; again check the current Free Tier page.

3. **Short Free Trials**
   - **App Runner:** Sometimes offers a trial (e.g. first few months). Check [App Runner pricing](https://aws.amazon.com/apprunner/pricing/).

### Can you deploy this app for free?

- **Yes, in a limited way:**
  - **EC2:** Run one **t2.micro** (or t3.micro) 24/7 within 750 h/month. Install Docker, run your container. Free tier fits one small instance.
  - **ECS Fargate:** Use free tier vCPU/GB-hours if available; run a small task (e.g. 0.25 vCPU, 0.5 GB RAM). Enough for light testing.
  - **App Runner:** If a trial or free tier applies, you can run the app with minimal config.
- **Important:** Set **billing alerts** in AWS (e.g. alert at $1 or $5) so you don’t accidentally exceed free tier (e.g. by using a large instance or high traffic).

---

## Part 3: Deployment Options (Summary)

| Option | Difficulty | Free tier? | Best for |
|--------|------------|-----------|----------|
| **EC2 + Docker** | Medium | Yes (t2.micro) | Learning, full control, 24/7 free-tier server |
| **App Runner** | Easiest | Trial / check pricing | Get a URL quickly, no server management |
| **ECS Fargate** | Medium | Limited free tier | Production-like, scaling, no server management |

Below are step-by-step instructions for each.

---

## Part 4: Prerequisites (All Options)

1. **AWS Account**  
   - Sign up at [aws.amazon.com](https://aws.amazon.com).  
   - Enable MFA and set a billing alert (e.g. $5) in **Billing → Budgets**.

2. **AWS CLI**  
   - Install: [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).  
   - Configure: `aws configure` (Access Key, Secret Key, default region e.g. `us-east-1`).

3. **Docker**  
   - Installed and working locally (you already use it for this project).

4. **Project ready**  
   - In the project root you have: `Dockerfile`, `app/`, `requirements.txt`, and `kbest.pt`.  
   - Test locally first:
     ```bash
     docker build -t plate-api .
     docker run -p 8000:8000 plate-api
     ```
     Then open `http://localhost:8000/docs`.

---

## Part 5: Option A — Deploy with AWS App Runner (Easiest)

App Runner runs a container from ECR and gives you a URL. No EC2 or ECS to configure.

### Step 1: Create ECR repository

```bash
aws ecr create-repository --repository-name plate-api --region us-east-1
```

Note the **repositoryUri** (e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com/plate-api`).

### Step 2: Log Docker into ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
```

Replace `123456789` with your AWS account ID (or use the host from the repositoryUri).

### Step 3: Build and tag image

From your project root:

```bash
docker build -t plate-api .
docker tag plate-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/plate-api:latest
```

### Step 4: Push image to ECR

```bash
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/plate-api:latest
```

### Step 5: Create App Runner service (Console)

1. In AWS Console go to **App Runner** → **Create service**.
2. **Source:**  
   - Repository type: **Container registry**.  
   - Provider: **Amazon ECR**.  
   - Select the **plate-api** image and tag **latest**.  
   - Deployment trigger: **Manual** (or Automatic if you want redeploy on push).
3. **Configure service:**  
   - Service name: e.g. `plate-api`.  
   - CPU: 1 vCPU, Memory: 2 GB (YOLO + EasyOCR need a bit of RAM).  
   - Port: **8000**.  
   - Environment variables (optional): `YOLO_DEVICE=cpu`, `YOLO_CONF=0.25`.
4. **Create service.**  
   When it’s ready, App Runner gives you a URL like `https://xxxxx.us-east-1.awsapprunner.com`.

### Step 6: Test

```bash
curl -X POST "https://YOUR_APP_RUNNER_URL/predict?conf=0.25" -F "file=@demo_frame.jpg"
```

- **Cost:** Check [App Runner pricing](https://aws.amazon.com/apprunner/pricing/). Use a small instance and set a billing alert.

---

## Part 6: Option B — Deploy with ECS Fargate

You run your container as an ECS task on Fargate. No EC2 to manage.

### Steps 1–4: Same as App Runner

- Create ECR repo, login, build, tag, push (see Option A).

### Step 5: Create ECS cluster

```bash
aws ecs create-cluster --cluster-name plate-cluster --region us-east-1
```

### Step 6: Create task execution role

1. IAM → Roles → Create role.  
2. Trusted entity: **AWS service** → **Elastic Container Service** → **Elastic Container Service Task**.  
3. Attach policies: **AmazonECSTaskExecutionRolePolicy**.  
4. Name: e.g. `ecsTaskExecutionRole`.

### Step 7: Register task definition

Create a file `ecs-task-def.json` (replace account ID and region), or use the one in this repo: `ecs-task-def.json`.

```json
{
  "family": "plate-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "plate-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/plate-api:latest",
      "portMappings": [{ "containerPort": 8000 }],
      "environment": [
        { "name": "YOLO_DEVICE", "value": "cpu" },
        { "name": "YOLO_CONF", "value": "0.25" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/plate-api",
          "awslogs-region": "us-east-1"
        }
      }
    }
  ]
}
```

Create log group:

```bash
aws logs create-log-group --log-group-name /ecs/plate-api --region us-east-1
```

Register task:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-def.json
```

### Step 8: Create load balancer + ECS service (simplified)

For a single running task with a public URL you typically:

- Create an **Application Load Balancer (ALB)** in a **VPC** (default VPC is fine for learning).  
- Create an **ECS service** that runs the task and registers it with the ALB.  
- Point a **target group** to the task on port 8000.

This is more steps; it’s often easier to do “Create cluster → Create task definition → Run task” and then use **Public IP** of the task to hit `http://TASK_PUBLIC_IP:8000` for testing. For production you’d add ALB + optional domain. Full ALB setup is in [AWS ECS Fargate docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html).

### Step 9: Run task (quick test)

- ECS → Clusters → **plate-cluster** → Tasks → Run new task.  
- Launch type: Fargate, same task definition, default VPC, public subnets, assign public IP.  
- After the task is running, get its public IP and open `http://TASK_IP:8000/docs`.  
- **Cost:** Fargate bills per vCPU and GB-hour. Stay within free tier if available; otherwise a few cents per hour for 0.5 vCPU / 1 GB.

---

## Part 7: Option C — Deploy on EC2 (Free Tier Friendly)

Run one small server 24/7 and run your Docker container on it.

### Step 1: Launch EC2 instance

1. EC2 → Launch instance.  
2. Name: `plate-api`.  
3. AMI: **Amazon Linux 2023**.  
4. Instance type: **t2.micro** (free tier).  
5. Key pair: Create or select one; download `.pem`.  
6. Network: Default VPC, allow **SSH (22)** and **HTTP (80)** or **Custom TCP 8000** in security group.  
7. Launch.

### Step 2: Connect and install Docker

```bash
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP
```

Then on the instance:

```bash
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user
```

Log out and back in so `docker` group applies.

### Step 3: Run your app (from your machine: copy files or use ECR)

**Option 3a – Use ECR (recommended)**  
- Push image to ECR (same as Option A steps 1–4).  
- On EC2: install AWS CLI, run `aws ecr get-login-password ... | docker login ...`, then:

  ```bash
  docker pull YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/plate-api:latest
  docker run -d -p 8000:8000 --name plate-api YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/plate-api:latest
  ```

**Option 3b – Build on EC2**  
- Copy project (e.g. with `scp -i your-key.pem -r ./project ec2-user@IP:~/plate-api`).  
- On EC2: `cd ~/plate-api && docker build -t plate-api . && docker run -d -p 8000:8000 plate-api`.

### Step 4: Open port 8000

- EC2 → Security groups → your instance’s group → Edit inbound rules → Add **Custom TCP 8000**, Source **0.0.0.0/0** (or your IP for testing).

### Step 5: Test

- Browser: `http://YOUR_EC2_PUBLIC_IP:8000/docs`.  
- **Cost:** $0 if you stay within 750 h/month of t2.micro (one instance 24/7).

---

## Part 8: Summary Checklist

- [ ] AWS account created; billing alert set.  
- [ ] AWS CLI installed and `aws configure` done.  
- [ ] Docker image runs locally (`docker build` / `docker run`).  
- [ ] ECR repo created; image pushed (for App Runner, Fargate, or EC2).  
- [ ] Choose one: **App Runner** (easiest), **ECS Fargate** (scalable), or **EC2** (free tier, one server).  
- [ ] Deploy using the steps above.  
- [ ] Call `/health` and `/predict` (or `/predict/annotated`) to verify.  
- [ ] Monitor usage in **Billing** and **Cost Explorer** so you stay within free tier or budget.

---

## Part 9: Quick Reference – Your Project on AWS

| What | Where |
|------|--------|
| API code | `app/main.py`, `app/inference.py` |
| Docker build | `Dockerfile` in project root |
| Model file | `kbest.pt` (in image or via volume) |
| Port | 8000 |
| Endpoints | `GET /health`, `POST /predict`, `POST /predict/annotated` |
| Env vars | `YOLO_DEVICE`, `YOLO_CONF`, `MODEL_PATH` (optional) |

You do **not** need to purchase anything to start: use Free Tier (EC2 t2.micro or Fargate/App Runner free tiers if available) and set a billing alert. For production or heavier traffic, you’ll pay for compute and data transfer; the same Docker image runs the same way across EC2, Fargate, and App Runner.
