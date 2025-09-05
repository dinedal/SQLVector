# SQLVector Release Automation Makefile
# Requires: gh (GitHub CLI), python, build, twine

# Read version from VERSION file
VERSION := $(shell cat VERSION)
PYTHON := python3
DIST_DIR := dist
BUILD_DIR := build
EGG_INFO := sqlvector.egg-info

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: help
help: ## Show this help message
	@echo "SQLVector Release Automation"
	@echo "=========================="
	@echo ""
	@echo "Current Version: $(GREEN)$(VERSION)$(NC)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

.PHONY: check-tools
check-tools: ## Check if required tools are installed
	@echo "Checking required tools..."
	@command -v gh >/dev/null 2>&1 || { echo "$(RED)Error: GitHub CLI (gh) is not installed$(NC)"; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)Error: Python is not installed$(NC)"; exit 1; }
	@command -v twine >/dev/null 2>&1 || { echo "$(YELLOW)Warning: twine is not installed. Installing...$(NC)"; pip install twine; }
	@echo "$(GREEN)All required tools are installed$(NC)"

.PHONY: check-auth
check-auth: ## Check GitHub CLI authentication
	@echo "Checking GitHub authentication..."
	@gh auth status || { echo "$(RED)Error: Not authenticated with GitHub. Run 'gh auth login'$(NC)"; exit 1; }
	@echo "$(GREEN)GitHub authentication confirmed$(NC)"

.PHONY: clean
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(DIST_DIR) $(BUILD_DIR) $(EGG_INFO)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Build artifacts cleaned$(NC)"

.PHONY: test
test: ## Run tests
	@echo "Running tests..."
	@pytest tests/ -v || { echo "$(RED)Tests failed$(NC)"; exit 1; }
	@echo "$(GREEN)All tests passed$(NC)"

.PHONY: lint
lint: ## Run linting checks
	@echo "Running linting checks..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check sqlvector/; \
	else \
		echo "$(YELLOW)Ruff not installed, skipping lint$(NC)"; \
	fi

.PHONY: build
build: clean ## Build distribution packages
	@echo "Building distribution packages for version $(VERSION)..."
	@$(PYTHON) -m pip install --upgrade build >/dev/null 2>&1
	@$(PYTHON) -m build
	@echo "$(GREEN)Build complete$(NC)"
	@ls -lh $(DIST_DIR)/

.PHONY: check-dist
check-dist: ## Check distribution packages with twine
	@echo "Checking distribution packages..."
	@twine check $(DIST_DIR)/* || { echo "$(RED)Distribution check failed$(NC)"; exit 1; }
	@echo "$(GREEN)Distribution packages are valid$(NC)"

.PHONY: test-release
test-release: build check-dist ## Upload to TestPyPI
	@echo "$(YELLOW)Uploading to TestPyPI...$(NC)"
	@echo "This will use your TestPyPI token from ~/.pypirc or prompt for credentials"
	@twine upload --repository testpypi $(DIST_DIR)/* || { echo "$(RED)TestPyPI upload failed$(NC)"; exit 1; }
	@echo "$(GREEN)Successfully uploaded to TestPyPI$(NC)"
	@echo "Test installation with: pip install --index-url https://test.pypi.org/simple/ sqlvector"

.PHONY: release-pypi
release-pypi: build check-dist ## Upload to PyPI (manual)
	@echo "$(YELLOW)Uploading to PyPI...$(NC)"
	@echo "This will use your PyPI token from ~/.pypirc or prompt for credentials"
	@read -p "Are you sure you want to release version $(VERSION) to PyPI? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload $(DIST_DIR)/* || { echo "$(RED)PyPI upload failed$(NC)"; exit 1; }; \
		echo "$(GREEN)Successfully uploaded to PyPI$(NC)"; \
	else \
		echo "$(YELLOW)Release cancelled$(NC)"; \
	fi

.PHONY: tag
tag: ## Create and push git tag
	@echo "Creating git tag v$(VERSION)..."
	@git diff --quiet || { echo "$(RED)Error: Uncommitted changes detected$(NC)"; exit 1; }
	@git tag -a v$(VERSION) -m "Release version $(VERSION)" || { echo "$(RED)Tag already exists$(NC)"; exit 1; }
	@git push origin v$(VERSION)
	@echo "$(GREEN)Tag v$(VERSION) created and pushed$(NC)"

.PHONY: release-github
release-github: check-tools check-auth tag ## Create GitHub release (triggers GitHub Action)
	@echo "Creating GitHub release for v$(VERSION)..."
	@gh release create v$(VERSION) \
		--title "v$(VERSION)" \
		--notes "Release version $(VERSION)\n\nSee [CHANGELOG.md](https://github.com/dinedal/sqlvector/blob/main/CHANGELOG.md) for details." \
		--draft || { echo "$(RED)Failed to create GitHub release$(NC)"; exit 1; }
	@echo "$(GREEN)Draft release created$(NC)"
	@echo "$(YELLOW)Visit https://github.com/dinedal/sqlvector/releases to review and publish$(NC)"
	@echo "$(YELLOW)Publishing the release will trigger automatic PyPI deployment via GitHub Actions$(NC)"

.PHONY: release-github-auto
release-github-auto: check-tools check-auth tag ## Create and publish GitHub release (auto-triggers PyPI deployment)
	@echo "Creating and publishing GitHub release for v$(VERSION)..."
	@read -p "This will create tag v$(VERSION) and trigger automatic PyPI deployment. Continue? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		gh release create v$(VERSION) \
			--title "v$(VERSION) - Release" \
			--notes-file CHANGELOG.md \
			--target main || { echo "$(RED)Failed to create GitHub release$(NC)"; exit 1; }; \
		echo "$(GREEN)Release v$(VERSION) published$(NC)"; \
		echo "$(YELLOW)GitHub Actions will now deploy to PyPI automatically$(NC)"; \
		echo "Monitor progress at: https://github.com/dinedal/sqlvector/actions"; \
	else \
		echo "$(YELLOW)Release cancelled$(NC)"; \
	fi

.PHONY: sync-version
sync-version: ## Sync pyproject.toml with VERSION file
	@echo "Syncing version $(VERSION) to pyproject.toml..."
	@sed -i.bak 's/version = "[0-9]*\.[0-9]*\.[0-9]*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@echo "$(GREEN)Version synced to $(VERSION)$(NC)"

.PHONY: bump-version
bump-version: ## Bump version (specify TYPE=patch|minor|major)
	@if [ -z "$(TYPE)" ]; then \
		echo "$(RED)Error: TYPE not specified. Use: make bump-version TYPE=patch|minor|major$(NC)"; \
		exit 1; \
	fi
	@echo "Current version: $(VERSION)"
	@NEW_VERSION=$$(echo $(VERSION) | awk -F. -v type=$(TYPE) '{ \
		if (type == "patch") print $$1"."$$2"."$$3+1; \
		else if (type == "minor") print $$1"."$$2+1".0"; \
		else if (type == "major") print $$1+1".0.0"; \
	}'); \
	echo "New version: $$NEW_VERSION"; \
	echo $$NEW_VERSION > VERSION; \
	sed -i.bak 's/version = "[0-9]*\.[0-9]*\.[0-9]*"/version = "'$$NEW_VERSION'"/' pyproject.toml && rm pyproject.toml.bak; \
	echo "$(GREEN)Version bumped to $$NEW_VERSION$(NC)"; \
	echo "$(YELLOW)Don't forget to update CHANGELOG.md$(NC)"

.PHONY: status
status: ## Show release status
	@echo "Release Status for SQLVector"
	@echo "=========================="
	@echo "Current Version: $(GREEN)$(VERSION)$(NC)"
	@echo ""
	@echo "Git Status:"
	@git status --short || echo "  Clean working directory"
	@echo ""
	@echo "Latest Tags:"
	@git tag -l | tail -5
	@echo ""
	@echo "GitHub Releases:"
	@gh release list --limit 3 2>/dev/null || echo "  No releases yet"
	@echo ""
	@echo "PyPI Package:"
	@curl -s https://pypi.org/pypi/sqlvector/json 2>/dev/null | grep -o '"version":"[^"]*"' | head -1 || echo "  Not yet on PyPI"

.PHONY: setup-secrets
setup-secrets: check-auth ## Interactive setup for GitHub secrets
	@echo "Setting up GitHub Secrets for PyPI deployment"
	@echo "============================================="
	@echo ""
	@echo "You'll need:"
	@echo "  1. PyPI API token (from https://pypi.org/manage/account/token/)"
	@echo "  2. TestPyPI API token (from https://test.pypi.org/manage/account/token/)"
	@echo ""
	@read -p "Enter your PyPI API token (starts with pypi-): " pypi_token; \
	if [ -n "$$pypi_token" ]; then \
		echo "$$pypi_token" | gh secret set PYPI_API_TOKEN --repo dinedal/sqlvector; \
		echo "$(GREEN)PyPI token saved$(NC)"; \
	fi
	@read -p "Enter your TestPyPI API token (starts with pypi-): " test_token; \
	if [ -n "$$test_token" ]; then \
		echo "$$test_token" | gh secret set TEST_PYPI_API_TOKEN --repo dinedal/sqlvector; \
		echo "$(GREEN)TestPyPI token saved$(NC)"; \
	fi
	@echo "$(GREEN)Secrets configured successfully$(NC)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	@pip install -e ".[test,duckdb]"
	@pip install build twine pytest pytest-asyncio ruff
	@echo "$(GREEN)Development dependencies installed$(NC)"

.PHONY: install-gh
install-gh: ## Install GitHub CLI
	@echo "Installing GitHub CLI..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		brew install gh; \
	elif [ "$$(uname)" = "Linux" ]; then \
		curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg; \
		echo "deb [arch=$$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null; \
		sudo apt update && sudo apt install gh; \
	else \
		echo "$(RED)Please install GitHub CLI manually from https://cli.github.com$(NC)"; \
	fi

.PHONY: full-release
full-release: test build check-dist release-github-auto ## Complete release process (test, build, and release via GitHub)
	@echo "$(GREEN)Full release process complete!$(NC)"

.PHONY: quick-release
quick-release: release-github-auto ## Quick release via GitHub Actions (skips local tests)
	@echo "$(GREEN)Quick release initiated via GitHub Actions$(NC)"

# Default target
.DEFAULT_GOAL := help