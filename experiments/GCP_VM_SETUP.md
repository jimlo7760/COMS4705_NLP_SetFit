# Running the experiments on a Google Cloud VM

The code, stages and guardrails are the same as on Colab (see `COLAB_SETUP.md` for what
each stage does). Only the setup differs: no Google Drive and no notebook. The code
reaches the VM through your team's private GitHub repo, you type commands in an SSH
terminal, and you download the results at the end.

Time: about 20 minutes of one-time setup, then an estimated 20–35 minutes of GPU runs
for the whole study. The smoke stage prints the real seconds per update.

## 0. Before you start (once per project)

1. **Billing.** The project must be linked to the billing account that holds the course
   credits. Google has not allowed GPUs on Free Trial accounts; course credit accounts
   normally don't have that limit. If creating the VM says GPUs are not allowed, this is
   why.
2. **GPU quota.** Many new projects start with a GPU quota of 0. Go to
   *IAM & Admin → Quotas & system limits*, filter for **GPUs (all regions)**, and if the
   limit is 0, edit it to **1** (reason: "course project, one T4 for about an hour").
   Also check **NVIDIA T4 GPUs** for the region you'll use. Approval can take
   minutes or longer, so do this first.
3. Open *Compute Engine* once and enable the API if asked.

## 1. Put the code on GitHub (once, on a Mac)

The project folder is already a Git repository with a first commit; `.gitignore` keeps
`data/`, `results/`, caches and LaTeX build files out of it.

1. On github.com: **New repository** → any name (e.g. `coms4705-setfit-banking77`) →
   **Private** → don't add a README, .gitignore or license → **Create repository**.
2. Add your teammates: repo → *Settings → Collaborators*.
3. In the project folder on your Mac, using the URL GitHub shows on the new repo's page
   (SSH `git@github.com:...` or HTTPS `https://github.com/...`, whichever you already use
   with GitHub):

```
git remote add origin <URL>
git push -u origin main
```

## 2. Create the VM

*Compute Engine → VM instances → Create instance*

| Setting | Value |
|---|---|
| Name | `setfit-t4` |
| Region / zone | `us-central1` (Iowa), any zone. If creation fails with "not enough resources", try another zone or `us-east1`. |
| Machine configuration | **GPUs** tab → GPU type **NVIDIA T4**, number **1** → machine type **n1-standard-4** |
| Provisioning model | **Standard** |
| OS and storage → Boot disk → Change | Operating system **Deep Learning on Linux**; version: the newest one that mentions CUDA 12; size: keep what it suggests |
| Networking | Defaults. The VM needs its external IP to reach GitHub and download the data and model. No HTTP/HTTPS rules needed. |

If the page offers to install the NVIDIA driver automatically, accept. The page shows
an hourly price on the right; for T4 + n1-standard-4 it should be well under $1/hour.

## 3. Connect

Click **SSH** next to the VM. A terminal opens in your browser.

On the first login, a Deep Learning VM asks whether to install the NVIDIA driver:
type `y` and wait 1–2 minutes. Then check:

```
nvidia-smi
```

It should list a **Tesla T4**.

## 4. Clone the repo and install (once per VM)

The VM gets **read-only** access to the repo through a deploy key, so no password or
token ever goes on the VM.

1. Make a key on the VM and print its public half:

```
ssh-keygen -t ed25519 -N "" -C setfit-vm -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

2. On GitHub: repo → *Settings → Deploy keys → Add deploy key*. Title `setfit-vm`, paste
   the line that `cat` printed, leave **Allow write access** unticked, **Add key**.
3. Clone and set up (replace `OWNER/REPO`; type `yes` when asked whether to trust
   github.com):

```
git clone git@github.com:OWNER/REPO.git ~/coms4705_setfit
bash ~/coms4705_setfit/experiments/vm_setup.sh
```

`vm_setup.sh` checks the driver, installs `tmux` if needed, creates a Python
environment in `~/setfit-venv` with the packages from `requirements.txt`, prints the GPU
and library versions, and runs the offline tests. It should end with `8 passed`.
It is safe to run again.

Data and results are written inside the clone (`~/coms4705_setfit/data/` and
`~/coms4705_setfit/results/`). Git ignores both, so `git pull` never touches them.

## 5. Run the stages

Start a `tmux` session first. Runs then keep going if the browser tab or SSH window
closes. To get back to them, click SSH again and run `tmux attach -t setfit`. To leave
on purpose without stopping anything, press `Ctrl-b`, then `d`.

As a safety net against forgetting to stop the VM, also schedule a shutdown. The VM
then stops itself after 2 hours, keeping its disk and results. `sudo shutdown -c`
cancels it if you need longer.

```
tmux new -s setfit
sudo shutdown -h +120
cd ~/coms4705_setfit/experiments
```

Then run one stage at a time, in this order:

```
./vm_run.sh --stage smoke
./vm_run.sh --stage frozen
./vm_run.sh --stage tune --shots 8
./vm_run.sh --stage tune --shots 32
./vm_run.sh --stage collect
```

**Stop here and do the CE check (below).** Then finish:

```
./vm_run.sh --stage select && ./vm_run.sh --stage final --shots 8 && ./vm_run.sh --stage final --shots 32 && ./vm_run.sh --stage collect
```

- Every run is saved when it finishes. After an interruption, re-run the same stage:
  finished runs are skipped.
- Each command also writes a log to `~/coms4705_setfit/results/logs/`.
- Run only one stage at a time on this VM (one GPU, one results folder).

### CE check (after tuning, before select)

The VM has no screen for the figure, so use the log instead:

```
grep "ce_k" ~/coms4705_setfit/results/logs/*_tune.log | tail -n 20
```

Each line reads `... step N: loss X val macro-F1 Y`. If CE's loss is still near **4.3**
(chance level, ln 77) at the last steps and its validation macro-F1 is far below
SetFit's, the CE head is barely learning. In that case, tune and lock CE with a separate
head learning rate **before** the `select` line above, and report this as a change since
the proposal:

```
./vm_run.sh --stage tune --methods ce --ce-head-lr 1e-3
./vm_run.sh --stage select --methods ce --ce-head-lr 1e-3
```

The plain `select` afterwards leaves CE as locked and only locks SetFit.

To look at the curves anyway, download `results/figures/tuning_curves.png` as in step 6.

## Changing the code

Edit on a Mac (with any teammate's clone), then push:

```
git add -A && git commit -m "Describe the fix" && git push
```

On the VM, pull and re-run the stage you were on:

```
cd ~/coms4705_setfit && git pull
```

- **Change code only up to the smoke stage.** That is when problems with the real model
  or GPU show up. Once `tune` has started, keep `setfit_study/` frozen: otherwise tuning
  and final runs come from different code, which weakens the comparison. Every run
  records a hash of the code (`code_fingerprint` in `results_log.csv`), so a mid-study
  change is visible afterwards. If one is unavoidable, re-run the affected runs with
  `--force` and mention it in the report.
- **Don't edit files on the VM.** If you did, `git pull` will refuse to overwrite them.
  Make the same change on the Mac and push it, then drop the VM copy with
  `git restore <file>` and pull again.

## 6. Bring the results back

```
tar czf ~/results.tar.gz -C ~/coms4705_setfit results
realpath ~/results.tar.gz
```

In the SSH window, click **Download file** and paste the path that `realpath` printed.
On a Mac, double-click the file to unpack it. Everything for the milestone is inside
(`table_main.tex`, `summary.csv`, `results_log.csv`, `figures/`, `preds/`, `logs/`).
The list in `COLAB_SETUP.md` ("What to bring back") applies unchanged.

If you unpack it inside the project folder, Git ignores it (`/results/` is in
`.gitignore`).

## 7. Stop paying

*VM instances → ⋮ → **Stop*** as soon as you're done. A running VM is billed every hour,
including the GPU, even when idle. A stopped VM costs only its disk (a few dollars a
month). Once the results are downloaded and backed up, **delete** the VM, and remove its
deploy key on GitHub (*Settings → Deploy keys*).

## Without GitHub

You can upload the code as one file instead. Build it in the project folder on the Mac:

```
COPYFILE_DISABLE=1 tar --no-xattrs --exclude __pycache__ --exclude .pytest_cache --exclude .DS_Store -czf experiments.tar.gz experiments
```

In the SSH window, click **Upload file** and choose `experiments.tar.gz`, then:

```
mkdir -p ~/coms4705_setfit
tar xzf ~/experiments.tar.gz -C ~/coms4705_setfit
bash ~/coms4705_setfit/experiments/vm_setup.sh
```

After a code change, rebuild, upload and unpack again; `data/` and `results/` are not
touched.

## If something fails

- **Quota error when creating the VM:** step 0.2.
- **"Does not have enough resources available":** pick another zone (or region) and create again.
- **`git clone` says "Permission denied (publickey)":** the deploy key isn't on the repo
  yet, or was pasted into your account's SSH keys instead of the repo's deploy keys.
- **`git: command not found`:** `sudo apt-get install -y git`.
- **`nvidia-smi` fails:** run `sudo /opt/deeplearning/install-driver.sh`, then `vm_setup.sh` again.
- **"PyTorch is installed but cannot see the GPU":** the driver is older than the
  PyTorch build needs; create the VM from the newest Deep Learning on Linux version.
- **Data audit or SHA-256 mismatch, or a crashed run:** same as in `COLAB_SETUP.md`.
