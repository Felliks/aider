from jinja2 import Environment, FileSystemLoader
from contextlib import contextmanager
from aider.io import InputOutput
from aider.coders import Coder
from aider.models import Model
from functools import wraps
from core.storage import image_generator
import aiohttp
import shutil
import asyncio
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('aider_content')

# Docker container root directory
ROOT_DIR = "/code"


def get_templates_dir():
    """
    Get absolute path to the templates directory.

    Returns:
        str: Absolute path to templates directory
    """
    templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    logger.info(f"Templates directory: {templates_dir}")
    return templates_dir


class SecurityError(Exception):
    """Exception raised for security violations in the landing generation process."""
    pass


class TimeoutError(Exception):
    """Exception raised when a function execution times out."""
    pass


def timeout(seconds):
    """
    Decorator that adds a timeout to a function.

    Args:
        seconds: The timeout in seconds

    Raises:
        TimeoutError: If the function execution exceeds the timeout
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        return wrapper

    return decorator


async def download_asset(url: str, path: str):
    """Download asset from URL to specified path"""
    logger.info(f"Downloading asset from {url} to {path}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(content)
                logger.info(f"Successfully downloaded asset to {path}")
            else:
                logger.error(f"Failed to download asset: {response.status}")


async def prepare_landing_dirs(landing: dict) -> str:
    """Prepare landing directories and download assets"""
    working_dir = f"/code/workspace/landings/{landing['id']}"
    logger.info(f"Preparing landing directory: {working_dir}")
    os.makedirs(working_dir, exist_ok=True)

    # Download assets
    assets_dir = os.path.join(working_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    for asset in landing.get('assets', []):
        original_name = asset.get('originalName')
        if original_name and asset.get('url'):
            dest_path = os.path.join(assets_dir, original_name)
            logger.info(f"Scheduling download of asset {original_name}")
            await download_asset(asset['url'], dest_path)

    return working_dir


def create_design_rules_content(landing: dict) -> str:
    """Create design rules content for task.md"""
    design_rules = []
    for i, rule in enumerate(landing.get('rules', [])):
        design_rules.append(f"{i + 1}. {rule['rule']}")

    return '\n'.join(design_rules)


def generate_task_md(landing: dict, working_dir: str) -> str:
    """
    Generate CREATE_TASK.md file for aider using Jinja2 templates

    Args:
        landing: Landing page data
        working_dir: Working directory path

    Returns:
        Path to the generated CREATE_TASK.md file
    """
    logger.info(f"Generating CREATE_TASK.md in {working_dir}")
    # Set up Jinja2 environment with absolute path
    templates_dir = get_templates_dir()
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('CREATE_TASK.md')

    # Prepare context with current datetime
    current_datetime = time.strftime("%m/%d/%Y, %H:%M:%S")
    context = {
        'landing'         : landing,
        'current_datetime': current_datetime,
        'assets'          : [a.get('originalName') for a in landing.get('assets', [])]
    }

    # Render template with context
    task_content = template.render(**context)

    # Write to file
    task_file_path = os.path.join(working_dir, 'CREATE_TASK.md')
    with open(task_file_path, 'w', encoding='utf-8') as f:
        f.write(task_content)
    logger.info(f"Generated CREATE_TASK.md successfully")

    return task_file_path


def generate_index_html(landing: dict, working_dir: str) -> str:
    # Set up Jinja2 environment with absolute path
    templates_dir = get_templates_dir()
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('index.html')

    # Render template with context
    index_content = template.render(landing=landing)

    # Write to file
    index_file_path = os.path.join(working_dir, 'index.html')
    with open(index_file_path, 'w', encoding='utf-8') as f:
        f.write(index_content)

    return index_file_path


def generate_style_css(landing: dict, working_dir: str) -> str:
    # Set up Jinja2 environment with absolute path
    templates_dir = get_templates_dir()
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('style.css')

    # Render template with landing object directly
    style_content = template.render(landing=landing)

    # Write to file
    style_file_path = os.path.join(working_dir, 'style.css')
    with open(style_file_path, 'w', encoding='utf-8') as f:
        f.write(style_content)

    return style_file_path


def generate_script_js(landing: dict, working_dir: str) -> str:
    # Set up Jinja2 environment with absolute path
    templates_dir = get_templates_dir()
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template('script.js')

    # Render template with landing object directly
    script_content = template.render(landing=landing)

    # Write to file
    script_file_path = os.path.join(working_dir, 'script.js')
    with open(script_file_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_file_path


@contextmanager
def temp_working_directory(working_dir):
    """
    Context manager for temporarily changing the working directory and returning back to ROOT_DIR.

    Args:
        working_dir: The directory to change to

    Yields:
        None
    """
    logger.info(f"Changing to working directory: {working_dir}")
    try:
        current_dir = os.getcwd()
        logger.info(f"Current directory before change: {current_dir}")
        os.chdir(working_dir)
        yield
    finally:
        logger.info(f"Changing back to root directory: {ROOT_DIR}")
        os.chdir(ROOT_DIR)


@timeout(1800)  # 30 minute timeout
async def run_aider_python_api_essential(working_dir: str, landing: dict):
    """Use aider Python API to run the code generation with Essential quality."""
    logger.info(f"Starting Aider Python API execution in {working_dir} with Essential quality")

    # Validate working directory is in the safe path
    safe_directory = os.path.join(ROOT_DIR, "workspace/landings/")
    absolute_working_dir = os.path.abspath(working_dir)

    logger.info(f"Validating working directory: {absolute_working_dir}")
    if not absolute_working_dir.startswith(safe_directory):
        logger.error(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")
        raise SecurityError(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")

    # Extract the landing_id from the path for additional security check
    landing_id = str(landing.get('id'))
    expected_path = os.path.join(safe_directory, landing_id)

    if absolute_working_dir != expected_path:
        logger.error(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")
        raise SecurityError(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")

    cost_usd = 0
    start_time = time.time()

    # Prepare file list to work with - include assets
    asset_files = [f"assets/{a['originalName']}" for a in landing.get('assets', [])]
    fnames = ['index.html', 'style.css', 'script.js']
    logger.info(f"Files to be processed: {', '.join(fnames)}")

    # Use the context manager to temporarily change to the working directory
    with temp_working_directory(working_dir):
        coder = Coder.create(
            main_model=Model('claude-3-5-haiku-20241022'),  # Standard LLM
            fnames=fnames,
            io=InputOutput(yes=True),  # Auto-confirm all actions
            use_git=False,
            auto_lint=True,
            detect_urls=False,
            # Langfuse parameters
            use_langfuse=True,
            langfuse_user_id=str(landing['user']['id']),
            langfuse_session_id=f"Landing #{landing['id']}",
            langfuse_metadata={
                "landing_id": landing['id'],
                "entity_type": "landing",
                "quality": landing.get('quality', 'standard'),
                "language": landing.get('language', 'en')
            },
            langfuse_tags=["landing", f"quality-{landing.get('quality', 'standard')}"]
        )

        coder.run('/read-only CREATE_TASK.md')

        # Mark asset files as read-only
        for asset_file in asset_files:
            if os.path.exists(asset_file):
                logger.info(f"Marking asset as read-only: {asset_file}")
                coder.run(f'/read-only "{asset_file}"')

        # Sort sections by position
        sections = sorted(landing['sections'], key=lambda x: x['position'])

        # Process each section individually
        for section in sections:
            section_type = "header" if section['position'] == 0 else "footer" if section == sections[-1] else "section"
            section_slug = section['slug']
            section_title = section.get('title', '')

            logger.info(f"Processing {section_type}: {section_slug} - {section_title}")

            # Create section-specific instructions
            section_instructions = f"""
I need you to implement the {section_type} with ID "{section_slug}" and title "{section_title}" in the index.html file.

Look at the index.html file and find the comment: <!-- {section_type.upper()}_TAG: {section_slug} -->

Replace this comment with the proper HTML implementation for this {section_type}.
Also add any necessary CSS to style.css and JavaScript to script.js.

Refer to TASK.md for detailed requirements about this specific {section_type}.
Refer to AIDER.md for general guidelines on landing page development.

This is section {section['position'] + 1} of {len(sections)}.
"""

            # Run aider for this specific section
            coder.run(section_instructions)

            # Give a short pause between sections to avoid rate limiting
            await asyncio.sleep(2)

        # Get token usage and cost directly from the coder object
        logger.info("Getting token usage information")
        token_count = coder.message_tokens_sent + coder.message_tokens_received
        cost_usd = coder.total_cost
        logger.info(f"Extracted cost: ${cost_usd:.6f}, token count: {token_count}")

        # Check if files were created/updated in the working directory
        logger.info("Verifying output files in working directory...")
        if not os.path.exists('index.html'):
            logger.error(f"index.html was not found in working directory")
            raise FileNotFoundError(f"index.html was not found in working directory")

        if not os.path.exists('style.css'):
            logger.error(f"style.css was not found in working directory")
            raise FileNotFoundError(f"style.css was not found in working directory")

        if not os.path.exists('script.js'):
            logger.error(f"script.js was not found in working directory")
            raise FileNotFoundError(f"script.js was not found in working directory")

    # Note: Now we're back in the ROOT_DIR
    end_time = time.time()
    duration_seconds = end_time - start_time
    logger.info(f"Execution took {duration_seconds:.2f} seconds")

    # Token count and cost are already set from the coder object

    # Add structured logging for metrics tracking
    logger.info(
        {
            "metric"          : "aider_generation",
            "token_count"     : token_count,
            "cost_usd"        : cost_usd,
            "duration_seconds": duration_seconds
        }
    )

    logger.info(f"Estimated cost: ${cost_usd:.6f}")

    # Final verification of files (using absolute paths)
    index_path = os.path.join(working_dir, 'index.html')
    style_path = os.path.join(working_dir, 'style.css')
    script_path = os.path.join(working_dir, 'script.js')

    logger.info("Verifying output files with absolute paths...")
    if not os.path.exists(index_path):
        logger.error(f"index.html was not found at {index_path}")
        raise FileNotFoundError(f"index.html was not found at {index_path}")

    if not os.path.exists(style_path):
        logger.error(f"style.css was not found at {style_path}")
        raise FileNotFoundError(f"style.css was not found at {style_path}")

    if not os.path.exists(script_path):
        logger.error(f"script.js was not found at {script_path}")
        raise FileNotFoundError(f"script.js was not found at {script_path}")

    logger.info(f"Aider Python API execution completed with cost ${cost_usd:.6f}")
    return cost_usd


@timeout(1800)  # 30 minute timeout
async def run_aider_python_api_expert(working_dir: str, landing: dict):
    """Use aider Python API to run the code generation with Expert quality."""
    logger.info(f"Starting Aider Python API execution in {working_dir} with Expert quality")

    # Validate working directory is in the safe path
    safe_directory = os.path.join(ROOT_DIR, "workspace/landings/")
    absolute_working_dir = os.path.abspath(working_dir)

    logger.info(f"Validating working directory: {absolute_working_dir}")
    if not absolute_working_dir.startswith(safe_directory):
        logger.error(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")
        raise SecurityError(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")

    # Extract the landing_id from the path for additional security check
    landing_id = str(landing.get('id'))
    expected_path = os.path.join(safe_directory, landing_id)

    if absolute_working_dir != expected_path:
        logger.error(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")
        raise SecurityError(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")

    cost_usd = 0
    start_time = time.time()

    # Dynamic import aider to avoid importing it in normal operation
    # when it might not be installed yet
    logger.info("Importing Aider dependencies")

    # Prepare file list to work with - include assets
    asset_files = [f"assets/{a['originalName']}" for a in landing.get('assets', [])]
    fnames = ['index.html', 'style.css', 'script.js']
    logger.info(f"Files to be processed: {', '.join(fnames)}")

    # Use the context manager to temporarily change to the working directory
    with temp_working_directory(working_dir):
        coder = Coder.create(
            main_model=Model('claude-3-7-sonnet-latest'),  # Premium LLM
            fnames=fnames,
            io=InputOutput(yes=True),  # Auto-confirm all actions
            use_git=False,
            auto_lint=True,
            detect_urls=False,
            # Langfuse parameters
            use_langfuse=True,
            langfuse_user_id=str(landing['user']['id']),
            langfuse_session_id=f"Landing #{landing['id']}",
            langfuse_metadata={
                "landing_id": landing['id'],
                "entity_type": "landing",
                "quality": landing.get('quality', 'standard'),
                "language": landing.get('language', 'en')
            },
            langfuse_tags=["landing", f"quality-{landing.get('quality', 'standard')}"]
        )

        coder.run('/read-only CREATE_TASK.md')

        # Mark asset files as read-only
        for asset_file in asset_files:
            if os.path.exists(asset_file):
                logger.info(f"Marking asset as read-only: {asset_file}")
                coder.run(f'/read-only {asset_file}')

        # Then give the simple instruction to follow it
        coder.run("Please implement the full landing page step by step according to the instructions in the CREATE_TASK.md file.")

        # Get token usage and cost directly from the coder object
        logger.info("Getting token usage information")
        token_count = coder.message_tokens_sent + coder.message_tokens_received
        cost_usd = coder.total_cost
        logger.info(f"Extracted cost: ${cost_usd:.6f}, token count: {token_count}")

        # Check if files were created/updated in the working directory
        logger.info("Verifying output files in working directory...")
        if not os.path.exists('index.html'):
            logger.error(f"index.html was not found in working directory")
            raise FileNotFoundError(f"index.html was not found in working directory")

        if not os.path.exists('style.css'):
            logger.error(f"style.css was not found in working directory")
            raise FileNotFoundError(f"style.css was not found in working directory")

        if not os.path.exists('script.js'):
            logger.error(f"script.js was not found in working directory")
            raise FileNotFoundError(f"script.js was not found in working directory")

    # Note: Now we're back in the ROOT_DIR
    end_time = time.time()
    duration_seconds = end_time - start_time
    logger.info(f"Execution took {duration_seconds:.2f} seconds")

    # Token count and cost are already set from the coder object

    # Add structured logging for metrics tracking
    logger.info(
        {
            "metric"          : "aider_generation",
            "token_count"     : token_count,
            "cost_usd"        : cost_usd,
            "duration_seconds": duration_seconds
        }
    )

    logger.info(f"Estimated cost: ${cost_usd:.6f}")

    # Final verification of files (using absolute paths)
    index_path = os.path.join(working_dir, 'index.html')
    style_path = os.path.join(working_dir, 'style.css')
    script_path = os.path.join(working_dir, 'script.js')

    logger.info("Verifying output files with absolute paths...")
    if not os.path.exists(index_path):
        logger.error(f"index.html was not found at {index_path}")
        raise FileNotFoundError(f"index.html was not found at {index_path}")

    if not os.path.exists(style_path):
        logger.error(f"style.css was not found at {style_path}")
        raise FileNotFoundError(f"style.css was not found at {style_path}")

    if not os.path.exists(script_path):
        logger.error(f"script.js was not found at {script_path}")
        raise FileNotFoundError(f"script.js was not found at {script_path}")

    logger.info(f"Aider Python API execution completed with cost ${cost_usd:.6f}")
    return cost_usd


@timeout(1800)  # 30 minute timeout
async def run_aider_python_api_master(working_dir: str, landing: dict):
    """Use aider Python API to run the code generation with Expert quality."""
    logger.info(f"Starting Aider Python API execution in {working_dir} with Master quality")

    # Validate working directory is in the safe path
    safe_directory = os.path.join(ROOT_DIR, "workspace/landings/")
    absolute_working_dir = os.path.abspath(working_dir)

    logger.info(f"Validating working directory: {absolute_working_dir}")
    if not absolute_working_dir.startswith(safe_directory):
        logger.error(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")
        raise SecurityError(f"Security violation: Working directory {absolute_working_dir} is outside of the allowed directory {safe_directory}")

    # Extract the landing_id from the path for additional security check
    landing_id = str(landing.get('id'))
    expected_path = os.path.join(safe_directory, landing_id)

    if absolute_working_dir != expected_path:
        logger.error(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")
        raise SecurityError(f"Security violation: Invalid working directory structure: {absolute_working_dir}, expected format: {expected_path}")

    cost_usd = 0
    start_time = time.time()

    # Dynamic import aider to avoid importing it in normal operation
    # when it might not be installed yet
    logger.info("Importing Aider dependencies")

    # Prepare file list to work with - include assets
    asset_files = [f"assets/{a['originalName']}" for a in landing.get('assets', [])]
    fnames = ['index.html', 'style.css', 'script.js']
    logger.info(f"Files to be processed: {', '.join(fnames)}")

    # Use the context manager to temporarily change to the working directory
    with temp_working_directory(working_dir):
        coder = Coder.create(
            main_model=Model('claude-opus-4-20250514'),  # Standard LLM
            fnames=fnames,
            io=InputOutput(yes=True),  # Auto-confirm all actions
            use_git=False,
            auto_lint=True,
            detect_urls=False,
            # Langfuse parameters
            use_langfuse=True,
            langfuse_user_id=str(landing['user']['id']),
            langfuse_session_id=f"Landing #{landing['id']}",
            langfuse_metadata={
                "landing_id": landing['id'],
                "entity_type": "landing",
                "quality": landing.get('quality', 'standard'),
                "language": landing.get('language', 'en')
            },
            langfuse_tags=["landing", f"quality-{landing.get('quality', 'standard')}"]
        )

        coder.run('/read-only CREATE_TASK.md')

        # Mark asset files as read-only
        for asset_file in asset_files:
            if os.path.exists(asset_file):
                logger.info(f"Marking asset as read-only: {asset_file}")
                coder.run(f'/read-only {asset_file}')

        # Then give the simple instruction to follow it
        coder.run("Please implement the full landing page step by step according to the instructions in the CREATE_TASK.md file.")

        # Get token usage and cost directly from the coder object
        logger.info("Getting token usage information")
        token_count = coder.message_tokens_sent + coder.message_tokens_received
        cost_usd = coder.total_cost
        logger.info(f"Extracted cost: ${cost_usd:.6f}, token count: {token_count}")

        # Check if files were created/updated in the working directory
        logger.info("Verifying output files in working directory...")
        if not os.path.exists('index.html'):
            logger.error(f"index.html was not found in working directory")
            raise FileNotFoundError(f"index.html was not found in working directory")

        if not os.path.exists('style.css'):
            logger.error(f"style.css was not found in working directory")
            raise FileNotFoundError(f"style.css was not found in working directory")

        if not os.path.exists('script.js'):
            logger.error(f"script.js was not found in working directory")
            raise FileNotFoundError(f"script.js was not found in working directory")

    # Note: Now we're back in the ROOT_DIR
    end_time = time.time()
    duration_seconds = end_time - start_time
    logger.info(f"Execution took {duration_seconds:.2f} seconds")

    # Token count and cost are already set from the coder object

    # Add structured logging for metrics tracking
    logger.info(
        {
            "metric"          : "aider_generation",
            "token_count"     : token_count,
            "cost_usd"        : cost_usd,
            "duration_seconds": duration_seconds
        }
    )

    logger.info(f"Estimated cost: ${cost_usd:.6f}")

    # Final verification of files (using absolute paths)
    index_path = os.path.join(working_dir, 'index.html')
    style_path = os.path.join(working_dir, 'style.css')
    script_path = os.path.join(working_dir, 'script.js')

    logger.info("Verifying output files with absolute paths...")
    if not os.path.exists(index_path):
        logger.error(f"index.html was not found at {index_path}")
        raise FileNotFoundError(f"index.html was not found at {index_path}")

    if not os.path.exists(style_path):
        logger.error(f"style.css was not found at {style_path}")
        raise FileNotFoundError(f"style.css was not found at {style_path}")

    if not os.path.exists(script_path):
        logger.error(f"script.js was not found at {script_path}")
        raise FileNotFoundError(f"script.js was not found at {script_path}")

    logger.info(f"Aider Python API execution completed with cost ${cost_usd:.6f}")
    return cost_usd


async def replace_asset_paths(content: str, landing: dict) -> str:
    """Replace local asset paths with CDN URLs in generated content"""
    logger.info("Replacing asset paths with CDN URLs")
    for asset in landing.get('assets', []):
        if asset.get('originalName') and asset.get('url'):
            local_path = f'assets/{asset["originalName"]}'
            cdn_url = asset['url']
            content = content.replace(local_path, cdn_url)
    return content


async def generate(landing: dict) -> tuple:
    logger.info(f"Starting landing page generation with Aider for landing ID: {landing['id']}")
    # Create necessary directories
    working_dir = await prepare_landing_dirs(landing)

    # Generate template files
    logger.info("Generating template files")
    task_path = generate_task_md(landing, working_dir)
    index_path = generate_index_html(landing, working_dir)
    style_path = generate_style_css(landing, working_dir)
    script_path = generate_script_js(landing, working_dir)

    # Try to run aider using Python API, fall back to shell command if needed
    logger.info(f"Attempting to run Aider with Python API using quality: {landing['quality']}")
    if landing['quality'] == 'master':
        cost_usd = await run_aider_python_api_master(working_dir, landing)
    elif landing['quality'] == 'expert':
        cost_usd = await run_aider_python_api_expert(working_dir, landing)
    else:
        cost_usd = await run_aider_python_api_essential(working_dir, landing)
    logger.info("Aider execution completed, processing results")

    # Read the full HTML content from index.html
    logger.info(f"Reading generated HTML from {index_path}")
    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Replace privacy and terms template tags with actual content
    logger.info("Checking for privacy and terms template tags")
    nl2br = lambda x: x.replace('\r\n', '<br>').replace('\n', '<br>')

    privacy_policy = landing.get('privacyPolicy') or ''
    privacy_policy = privacy_policy.strip()
    if len(privacy_policy):
        logger.info("Replacing {privacy} tag with privacy policy content (with nl2br)")
        html_content = html_content.replace('{privacy}', nl2br(privacy_policy))

    terms_conditions = landing.get('termsAndConditions') or ''
    terms_conditions = terms_conditions.strip()
    if len(terms_conditions):
        logger.info("Replacing {terms} tag with terms and conditions content (with nl2br)")
        html_content = html_content.replace('{terms}', nl2br(terms_conditions))

    # Replace asset paths with CDN URLs
    logger.info("Replacing asset paths with CDN URLs")
    html_content = await replace_asset_paths(html_content, landing)

    # Generate images in the HTML content
    logger.info("Generate images in HTML content")
    html_content = await image_generator.replace_images_in_html(
        html_content,
        landing.get('imageModel', image_generator.MODEL_FLUX)
    )

    # Read style.css content
    logger.info(f"Reading generated CSS from {style_path}")
    with open(style_path, 'r', encoding='utf-8') as f:
        style_content = f.read()

    # Replace asset paths in CSS
    style_content = await replace_asset_paths(style_content, landing)

    # Generate images in the CSS content
    logger.info("Generate images in CSS content")
    style_content = await image_generator.replace_images_in_html(
        style_content,
        landing.get('imageModel', image_generator.MODEL_FLUX)
    )

    # Read script.js content
    logger.info(f"Reading generated JS from {script_path}")
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Replace asset paths in JS
    script_content = await replace_asset_paths(script_content, landing)

    # Generate images in the JS content
    logger.info("Generate images in JS content")
    script_content = await image_generator.replace_images_in_html(
        script_content,
        landing.get('imageModel', image_generator.MODEL_NOVA)
    )

    # Create the result tuple
    result = (html_content, style_content, script_content, cost_usd)

    # Clean up working directory
    logger.info(f"Cleaning up working directory: {working_dir}")
    shutil.rmtree(working_dir)

    logger.info(f"Landing page generation completed successfully with cost ${cost_usd:.6f}")
    return result
