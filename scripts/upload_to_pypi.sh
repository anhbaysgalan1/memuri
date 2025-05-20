#!/bin/bash
# Upload Memuri package to PyPI with automatic version increment

set -e  # Exit on error

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Memuri PyPI Upload Tool${NC}"
echo "=============================="

# Check if required tools are installed
for cmd in poetry twine; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed.${NC}"
        if [ "$cmd" = "poetry" ]; then
            echo "Please install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
        elif [ "$cmd" = "twine" ]; then
            echo "Please install Twine: pip install twine"
        fi
        exit 1
    fi
done

# Make sure we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found.${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep "^version" pyproject.toml | cut -d'"' -f2)
echo -e "Current version: ${GREEN}${CURRENT_VERSION}${NC}"

# Parse version components
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Increment patch version by default
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"

# Ask for version increment type
echo -e "\nSelect version increment type:"
echo "1) Patch (${MAJOR}.${MINOR}.${NEW_PATCH}) - for bug fixes"
echo "2) Minor (${MAJOR}.$((MINOR + 1)).0) - for new features"
echo "3) Major ($((MAJOR + 1)).0.0) - for breaking changes"
echo "4) Custom (enter manually)"
read -p "Enter choice [1-4] (default: 1): " VERSION_CHOICE
echo

case $VERSION_CHOICE in
    2)
        NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
        ;;
    3)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    4)
        read -p "Enter custom version: " CUSTOM_VERSION
        if [[ ! $CUSTOM_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo -e "${RED}Error: Invalid version format. Use format: X.Y.Z${NC}"
            exit 1
        fi
        NEW_VERSION=$CUSTOM_VERSION
        ;;
    *)
        # Default to patch increment (option 1 or invalid input)
        ;;
esac

echo -e "Will update to: ${GREEN}${NEW_VERSION}${NC}"

# Update version in pyproject.toml
echo "Updating version in pyproject.toml..."
# Use different approach for macOS/BSD sed vs GNU sed
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS/BSD version
    sed -i '' "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
else
    # GNU/Linux version
    sed -i "s/^version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
fi

# Clean previous builds
echo -e "\n${GREEN}Cleaning previous builds...${NC}"
rm -rf dist/
rm -rf build/

# Run tests
echo -e "\n${GREEN}Running tests...${NC}"
poetry run pytest

# Build the package
echo -e "\n${GREEN}Building package ${NEW_VERSION}...${NC}"
poetry build

# Show build files
echo -e "\n${GREEN}Build complete. Package details:${NC}"
ls -lh dist/

# Confirm upload to PyPI
echo
read -p "Upload to PyPI? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}Upload canceled. You can upload later with:${NC}"
    echo "python -m twine upload dist/*"
    exit 0
fi

# Check if token is set
if [ -z "$PYPI_TOKEN" ]; then
    echo -e "\n${YELLOW}PYPI_TOKEN environment variable not set.${NC}"
    echo -e "You can either:"
    echo "1) Enter a token now (recommended)"
    echo "2) Use username/password authentication"
    read -p "Option [1-2]: " AUTH_OPTION
    
    if [ "$AUTH_OPTION" = "1" ]; then
        read -sp "Enter PyPI token: " PYPI_TOKEN
        echo
        AUTH_ARGS="--username __token__ --password $PYPI_TOKEN"
    else
        AUTH_ARGS=""  # Will prompt for username/password
    fi
else
    echo -e "\n${GREEN}Using PYPI_TOKEN from environment${NC}"
    AUTH_ARGS="--username __token__ --password $PYPI_TOKEN"
fi

# Upload to PyPI
echo -e "\n${GREEN}Uploading to PyPI...${NC}"
UPLOAD_SUCCESS=0

# Try upload with appropriate authentication
if [ -z "$AUTH_ARGS" ]; then
    python -m twine upload dist/* && UPLOAD_SUCCESS=1
else
    python -m twine upload dist/* $AUTH_ARGS && UPLOAD_SUCCESS=1
fi

# Create git commit and tag if upload was successful
if [ $UPLOAD_SUCCESS -eq 1 ]; then
    echo -e "\n${GREEN}Upload successful.${NC}"
    
    # Ask to commit and tag
    read -p "Create git commit and tag for v${NEW_VERSION}? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add pyproject.toml
        git commit -m "Bump version to ${NEW_VERSION}"
        git tag -a "v${NEW_VERSION}" -m "Release version ${NEW_VERSION}"
        
        echo -e "\n${GREEN}Git commit and tag created.${NC}"
        read -p "Push changes to remote repository? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push && git push --tags
            echo -e "\n${GREEN}Changes pushed to remote repository.${NC}"
        fi
    fi
else
    echo -e "\n${RED}Upload failed.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Done!${NC}" 