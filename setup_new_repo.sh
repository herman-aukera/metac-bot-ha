#!/bin/bash

# Script to set up a new repository for the tournament optimization system
# Repository name: metac-agent-agent

echo "Setting up new repository: metac-agent-agent"

# Step 1: Create a new directory for the new repository
echo "Step 1: Creating new repository directory..."
cd ..
mkdir metac-agent-agent
cd metac-agent-agent

# Step 2: Initialize new git repository
echo "Step 2: Initializing new git repository..."
git init
git branch -M main

# Step 3: Copy all files from the current project (excluding .git)
echo "Step 3: Copying project files..."
rsync -av --exclude='.git' ../$(basename "$OLDPWD")/ ./

# Step 4: Create initial commit
echo "Step 4: Creating initial commit..."
git add .
git commit -m "Initial commit: Tournament Optimization System

- Complete tournament orchestration and integration layer
- Advanced forecasting pipeline with multi-agent ensemble
- Comprehensive error handling and resilience
- REST API and CLI interfaces
- Backward compatibility with existing main entry points
- Full test coverage and monitoring capabilities"

# Step 5: Instructions for GitHub setup
echo ""
echo "=========================================="
echo "Next steps to complete the setup:"
echo "=========================================="
echo ""
echo "1. Go to GitHub and create a new repository named 'metac-agent-agent'"
echo "   - Make it public or private as needed"
echo "   - Don't initialize with README, .gitignore, or license (we already have these)"
echo ""
echo "2. Add the remote origin and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/metac-agent-agent.git"
echo "   git push -u origin main"
echo ""
echo "3. Your new repository will be ready at:"
echo "   https://github.com/YOUR_USERNAME/metac-agent-agent"
echo ""
echo "Current directory: $(pwd)"
echo "Repository initialized successfully!"
