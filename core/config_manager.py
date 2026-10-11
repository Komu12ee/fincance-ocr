"""
CHiPPY Configuration Manager
Provides centralized, dynamic path and configuration management for all pipeline stages.

Usage:
    from config_manager import Config
    
    config = Config(stage='preprocessing')
    input_path = config.get_input_path()
    output_path = config.get_output_path()
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()


class Config:
    """Centralized configuration manager for CHiPPY pipeline."""
    
    # Stage definitions with default I/O paths
    STAGES = {
        'preprocessing': {
            'input_dir': 'input_pdfs',
            'output_dir': 'stage1_output',
            'stage_path': '01_preprocessing'
        },
        'optimization': {
            'input_dir': 'stage2_output',  # From preprocessing stage 2
            'output_dir': 'output',
            'stage_path': '02_optimization'
        },
        'chunking': {
            'input_dir': '../02_optimization/output',
            'output_dir': 'output',
            'stage_path': '03_chunking'
        },
        'embeddings': {
            'input_dir': 'data',
            'output_dir': 'db',
            'stage_path': '04_embeddings_and_kg'
        },
        'webui': {
            'input_dir': '../04_embeddings_and_kg/db',
            'output_dir': 'results',
            'stage_path': '05_webui'
        }
    }
    
    def __init__(self, stage: str = None, base_dir: str = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            stage: Pipeline stage ('preprocessing', 'optimization', 'chunking', 'embeddings', 'webui')
            base_dir: Base directory for CHiPPY (defaults to script's parent)
            **kwargs: Override specific paths/configs
        """
        self.stage = stage
        self.base_dir = Path(base_dir) if base_dir else self._get_base_dir()
        self.config_dict = kwargs
        
        # Load stage configuration if provided
        if stage and stage in self.STAGES:
            self.stage_config = self.STAGES[stage].copy()
        else:
            self.stage_config = {}
    
    @staticmethod
    def _get_base_dir() -> Path:
        """Determine CHiPPY base directory."""
        # Check environment variable first
        if env_base := os.getenv('CHIPPY_BASE_DIR'):
            return Path(env_base)
        
        # Check for common CHiPPY marker files
        current = Path.cwd()
        for _ in range(5):  # Look up to 5 levels
            if (current / 'README.md').exists() and (current / '01_preprocessing').exists():
                return current
            current = current.parent
        
        # Default to current directory
        return Path.cwd()
    
    def get_stage_dir(self) -> Path:
        """Get the stage directory."""
        if not self.stage:
            raise ValueError("Stage not set in config")
        
        stage_path = self.stage_config.get('stage_path', f'0X_{self.stage}')
        return self.base_dir / stage_path
    
    def get_input_path(self, custom: str = None, as_str: bool = False) -> Path:
        """
        Get input directory for current stage.
        
        Args:
            custom: Override input path
            as_str: Return as string instead of Path
            
        Returns:
            Path or string to input directory
        """
        if custom:
            path = Path(custom)
        else:
            # Try environment variable first
            env_key = f'CHIPPY_{self.stage.upper()}_INPUT'
            if env_input := os.getenv(env_key):
                path = Path(env_input)
            # Then config override
            elif override := self.config_dict.get('input_dir'):
                path = Path(override)
            # Then default
            else:
                input_dir = self.stage_config.get('input_dir', 'input')
                path = self.get_stage_dir() / input_dir
        
        # Make absolute if relative
        if not path.is_absolute():
            path = self.base_dir / path
        
        return str(path) if as_str else path
    
    def get_output_path(self, custom: str = None, as_str: bool = False) -> Path:
        """
        Get output directory for current stage.
        
        Args:
            custom: Override output path
            as_str: Return as string instead of Path
            
        Returns:
            Path or string to output directory
        """
        if custom:
            path = Path(custom)
        else:
            # Try environment variable first
            env_key = f'CHIPPY_{self.stage.upper()}_OUTPUT'
            if env_output := os.getenv(env_key):
                path = Path(env_output)
            # Then config override
            elif override := self.config_dict.get('output_dir'):
                path = Path(override)
            # Then default
            else:
                output_dir = self.stage_config.get('output_dir', 'output')
                path = self.get_stage_dir() / output_dir
        
        # Make absolute if relative
        if not path.is_absolute():
            path = self.base_dir / path
        
        return str(path) if as_str else path
    
    def get_data_path(self, filename: str = None, subdir: str = None, as_str: bool = False):
        """
        Get path to a data file.
        
        Args:
            filename: Specific file
            subdir: Subdirectory within data folder
            as_str: Return as string instead of Path
            
        Returns:
            Path or string to data location
        """
        data_dir = self.get_input_path() if self.stage != 'embeddings' else self.base_dir / '04_embeddings_and_kg' / 'data'
        
        if subdir:
            path = data_dir / subdir
        else:
            path = data_dir
        
        if filename:
            path = path / filename
        
        return str(path) if as_str else path
    
    def get_config_file(self, config_type: str = 'python') -> Path:
        """
        Get configuration file path for current stage.
        
        Args:
            config_type: Type of config ('python', 'yaml', 'json', 'env')
            
        Returns:
            Path to configuration file
        """
        stage_dir = self.get_stage_dir()
        
        if config_type == 'python':
            return stage_dir / 'config.py'
        elif config_type == 'yaml':
            return self.base_dir / f'{self.stage}_config.yaml'
        elif config_type == 'json':
            return self.base_dir / f'{self.stage}_config.json'
        elif config_type == 'env':
            return self.base_dir / '.env'
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def create_directories(self):
        """Create necessary input and output directories."""
        input_path = self.get_input_path()
        output_path = self.get_output_path()
        
        Path(input_path).mkdir(parents=True, exist_ok=True)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        return Path(input_path), Path(output_path)
    
    def validate_input_exists(self) -> bool:
        """Check if input path exists."""
        input_path = self.get_input_path()
        return Path(input_path).exists()
    
    def log_config(self):
        """Print current configuration (for debugging)."""
        print(f"\n{'='*60}")
        print(f"CHiPPY Configuration - Stage: {self.stage.upper()}")
        print(f"{'='*60}")
        print(f"Base Directory: {self.base_dir}")
        print(f"Stage Directory: {self.get_stage_dir()}")
        print(f"Input Path: {self.get_input_path()}")
        print(f"Output Path: {self.get_output_path()}")
        print(f"Input Exists: {self.validate_input_exists()}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def load_from_file(config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to config file (json, yaml, or env)
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_file)
        
        if config_file.suffix == '.json':
            with open(config_file) as f:
                return json.load(f)
        elif config_file.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_file) as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config")
        elif config_file.suffix == '.env' or config_file.name == '.env':
            load_dotenv(config_file)
            return dict(os.environ)
        else:
            raise ValueError(f"Unsupported config file: {config_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'stage': self.stage,
            'base_dir': str(self.base_dir),
            'input_path': str(self.get_input_path()),
            'output_path': str(self.get_output_path()),
            'stage_config': self.stage_config,
            'overrides': self.config_dict
        }


class PathManager:
    """Helper class for path operations."""
    
    @staticmethod
    def ensure_dirs(*paths):
        """Ensure multiple directories exist."""
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_files(directory: str, pattern: str = '*', recursive: bool = False):
        """Get files matching pattern."""
        dir_path = Path(directory)
        if recursive:
            return list(dir_path.rglob(pattern))
        return list(dir_path.glob(pattern))
    
    @staticmethod
    def get_relative_path(file_path: str, base_dir: str = None) -> Path:
        """Get relative path to file."""
        file_path = Path(file_path)
        base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        try:
            return file_path.relative_to(base_dir)
        except ValueError:
            return file_path
    
    @staticmethod
    def get_stem(file_path: str) -> str:
        """Get filename without extension."""
        return Path(file_path).stem
    
    @staticmethod
    def get_files_by_stage(config: Config, file_extension: str = None):
        """Get all files from input directory."""
        input_path = config.get_input_path()
        
        if file_extension:
            pattern = f'*{file_extension}'
        else:
            pattern = '*'
        
        return PathManager.get_files(input_path, pattern)


# Example usage and testing
if __name__ == '__main__':
    # Test configuration loading
    config = Config(stage='preprocessing')
    config.log_config()
    
    # Create directories
    config.create_directories()
    
    # Get paths
    print(f"Input: {config.get_input_path(as_str=True)}")
    print(f"Output: {config.get_output_path(as_str=True)}")
    print(f"Config as dict: {config.to_dict()}")
