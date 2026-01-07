# Working with GitHub Codespaces

This guide explains how to use GitHub Codespaces for developing the MTC Extraction Benchmark project.

## What is GitHub Codespaces?

GitHub Codespaces provides a complete, configurable development environment in the cloud. It uses the same devcontainer configuration as VS Code, so you get a consistent development experience whether working locally or in the cloud.

## Getting Started with Codespaces

### Creating a Codespace

1. **From GitHub Web Interface:**
   - Navigate to the repository: `https://github.com/GoSmarter-ai/mtc-extraction-benchmark`
   - Click the green "Code" button
   - Select the "Codespaces" tab
   - Click "Create codespace on main" (or your branch)

2. **From VS Code:**
   - Install the [GitHub Codespaces extension](https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces)
   - Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Type "Codespaces: Create New Codespace"
   - Select the repository and branch

3. **From GitHub CLI:**
   ```bash
   gh codespace create --repo GoSmarter-ai/mtc-extraction-benchmark
   ```

### Choosing Machine Type

GitHub Codespaces offers different machine types:

- **2-core** (default): Good for basic development and documentation
- **4-core**: Better for running smaller models and tests
- **8-core**: Recommended for training and evaluating ML models
- **16-core** or **32-core**: For intensive ML workloads

**For this project, we recommend:**
- 4-core minimum for development
- 8-core or higher for model training and evaluation

To change machine type:
1. Click on the gear icon in Codespaces
2. Select "Change Machine Type"
3. Choose your preferred configuration

## Working in Your Codespace

### Opening the Codespace

Once created, your Codespace will open in VS Code (web or desktop):
- All extensions from `devcontainer.json` will be installed
- Python environment will be configured
- All dependencies will be available

### Managing Files and Data

#### Uploading Data Files

For datasets and test documents:
1. Use the VS Code file explorer: Right-click → Upload
2. Or use `gh` CLI from terminal:
   ```bash
   # From local machine
   gh codespace cp ./local/data.zip remote:/workspace/data/
   ```

#### Downloading Results

```bash
# From Codespace to local
gh codespace cp remote:/workspace/results.zip ./local/
```

### Running Code

#### Python Scripts
```bash
# Run directly in terminal
python scripts/extract_certificates.py

# Or with specific environment
python -m src.models.evaluate
```

#### Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Access the notebook via the forwarded port (Codespaces will show a notification).

### Using GPU in Codespaces

GitHub Codespaces can provide GPU-enabled machines:

1. **Create GPU Codespace:**
   - Currently requires GitHub Enterprise or special access
   - Select a GPU-enabled machine type when creating

2. **Verify GPU availability:**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   nvidia-smi
   ```

**Note:** GPU Codespaces may have limited availability and higher costs.

## Codespace Management

### Stopping a Codespace

Codespaces automatically stop after 30 minutes of inactivity (default).

**Manual stop:**
- Web: Click the codespace name → "Stop codespace"
- CLI: `gh codespace stop -c CODESPACE_NAME`

**Important:** Stopped Codespaces retain all files and changes.

### Deleting a Codespace

When you're done with a Codespace:

1. **Via GitHub:**
   - Go to github.com/codespaces
   - Click "..." next to the codespace
   - Select "Delete"

2. **Via CLI:**
   ```bash
   gh codespace delete -c CODESPACE_NAME
   ```

### Viewing All Codespaces

```bash
# List all your codespaces
gh codespace list

# View in browser
open https://github.com/codespaces
```

## Advanced Features

### Port Forwarding

Your `devcontainer.json` already configures ports 8888 and 5000:

```json
"forwardPorts": [8888, 5000]
```

**Access forwarded ports:**
1. VS Code will show a notification when a port is detected
2. Go to "Ports" tab in VS Code
3. Click the local address to open in browser

**Make port public** (accessible without authentication):
- Right-click port → "Port Visibility" → "Public"

### Secrets and Environment Variables

For API keys and credentials:

1. **Codespace Secrets:**
   - Go to GitHub Settings → Codespaces → Secrets
   - Add secrets (e.g., `OPENAI_API_KEY`, `HF_TOKEN`)
   - These are available as environment variables

2. **Repository Secrets:**
   - Can be used if configured in the repository

3. **Using secrets in code:**
   ```python
   import os
   api_key = os.getenv('OPENAI_API_KEY')
   ```

### Personalizing Your Codespace

Create `~/.dotfiles` repository with your preferences:
- Shell configuration (`.bashrc`, `.zshrc`)
- Git configuration (`.gitconfig`)
- VS Code settings

GitHub will automatically apply these to new Codespaces.

### Rebuilding the Container

If you update `.devcontainer/`:

1. **Command Palette:** "Codespaces: Rebuild Container"
2. **Full rebuild:** "Codespaces: Full Rebuild Container"

## Cost Management

### Understanding Costs

Free tier includes:
- 120 core-hours/month for personal accounts
- 180 core-hours/month for Pro accounts

**Calculate usage:**
- 2-core machine for 1 hour = 2 core-hours
- 8-core machine for 1 hour = 8 core-hours

### Best Practices for Cost Savings

1. **Stop when not in use:** Don't leave Codespaces running
2. **Use appropriate machine size:** Don't use 32-core for simple tasks
3. **Set timeout:** Configure auto-stop timeout in settings
4. **Delete unused Codespaces:** Clean up old workspaces

### Monitoring Usage

Check your usage:
1. Go to [GitHub Billing](https://github.com/settings/billing)
2. View "Codespaces" section
3. See core-hours used and storage

## Troubleshooting

### Codespace is Slow

**Solutions:**
1. Upgrade machine type
2. Check if large file operations are running
3. Clear cache: `docker system prune`

### Package Installation Fails

**Solutions:**
1. Rebuild container: "Codespaces: Rebuild Container"
2. Check Dockerfile for syntax errors
3. Verify package names and versions

### Cannot Connect to Codespace

**Solutions:**
1. Check GitHub status: https://www.githubstatus.com/
2. Try opening in different browser
3. Use VS Code desktop instead of web

### Running Out of Storage

Each Codespace has limited storage (default 32GB).

**Check usage:**
```bash
df -h
du -sh * | sort -h
```

**Free up space:**
```bash
# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Remove pip cache
pip cache purge

# Remove Docker cache
docker system prune -a
```

## Workflow Examples

### Example 1: Quick Experiment

```bash
# Create Codespace (2-core is fine)
gh codespace create -r GoSmarter-ai/mtc-extraction-benchmark

# Connect
gh codespace ssh

# Run quick test
python scripts/test_ocr.py

# Stop when done
gh codespace stop
```

### Example 2: Model Training

```bash
# Create with more resources (8-core)
gh codespace create -r GoSmarter-ai/mtc-extraction-benchmark -m large

# Upload training data
gh codespace cp ./data/train.zip remote:/workspace/data/

# Connect and train
gh codespace ssh
python scripts/train_model.py

# Download results
gh codespace cp remote:/workspace/models/best_model.pt ./
```

### Example 3: Collaborative Development

1. Create a feature branch locally
2. Push to GitHub
3. Create Codespace on that branch
4. Make changes in Codespace
5. Commit and push from Codespace
6. Create PR from GitHub

## Integration with GitHub Models

Codespaces work seamlessly with GitHub Models API. See the [GitHub Models Integration Guide](./github-models-integration.md) for details.

## Tips and Tricks

1. **Pre-build containers:** Configure prebuild for faster Codespace startup
2. **Use dotfiles:** Customize your environment automatically
3. **SSH access:** Use `gh codespace ssh` for terminal-only access
4. **VS Code settings sync:** Enable Settings Sync for consistent experience
5. **Extensions:** All extensions from devcontainer are pre-installed

## Resources

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Codespaces Billing](https://docs.github.com/en/billing/managing-billing-for-github-codespaces)
- [DevContainer Specification](https://containers.dev/)

## Next Steps

- Learn about [DevContainer Iteration](./devcontainer-iteration.md)
- Explore [GitHub Models Integration](./github-models-integration.md)
