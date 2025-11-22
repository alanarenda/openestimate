import os
import glob
import json
import re
import textwrap
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from reportlab.lib import colors
from scipy.stats import norm, beta
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet


def get_variable_difficulty(var_name, variables):
    for key, var in variables.items():
        if var.get('variable') == var_name:
            if 'easy' in key:
                return 'easy'
            elif 'medium' in key:
                return 'medium'
            elif 'hard' in key:
                return 'hard'
            else:
                return 'base'



def get_variable_name(var_name, variables):
    for name, var in variables.items():
        if var['variable'] == var_name:
            return name


def draw_chat_bubble(c, x, y, width, height, color, text):
    c.setFillColor(color)
    c.roundRect(x, y, width, height, 10, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)

    # Create a style for LaTeX rendering
    styles = getSampleStyleSheet()
    style = styles['Normal']

    # Draw each line of text with LaTeX formatting
    for i, line in enumerate(text):
        try:
            # Use Paragraph to render LaTeX
            p = Paragraph(line, style)
            p.wrapOn(c, width - 20, height)
            p.drawOn(c, x + 10, y + height - 15 - (i * 12))
        except ValueError as e:
            print(f"Error rendering line {i}: {line}")
            print(f"Error: {e}")
            # Optionally, handle the error or skip the line


def prettify_chat_logs_to_pdf(elicited_priors, output_dir, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    margin = 50
    y_position = height - margin

    for var_name, info in elicited_priors.items():
        c.setFont("Helvetica-Bold", 12)
        # Wrap the title text to fit within the page width
        wrapped_title = textwrap.fill(f"Conversation for {var_name}", width=80)
        title_lines = wrapped_title.split('\n')
        title_height = 15 * len(title_lines)  # Calculate height based on wrapped lines

        # Draw each line of the wrapped title
        for line in title_lines:
            c.drawString(30, y_position, line)
            y_position -= 15  # Move down for the next line

        y_position -= 10  # Additional space after the title

        for message in info[var_name]['conversation']:
            role = message['role'].capitalize()
            content = message['content']
            bubble_color = colors.lightblue if role == 'User' else colors.lightgreen
            text = f"{role}: {content}"

            # Wrap text to fit within the bubble width
            wrapped_text = textwrap.wrap(text, width=80)
            bubble_height = 15 * len(wrapped_text) + 10

            # Check if there's enough space on the page for the bubble
            if y_position - bubble_height < margin:
                c.showPage()
                y_position = height - margin

            draw_chat_bubble(c, 30, y_position - bubble_height, width - 60, bubble_height, bubble_color, wrapped_text)
            y_position -= bubble_height + 20

        c.showPage()
    c.save()


def replace_placeholders(protocol_lines, all_vars):
    updated_lines = protocol_lines
    placeholders = re.findall(r'\{\{(.*?)\}\}', protocol_lines)
    for placeholder in placeholders:
        if placeholder in all_vars:  # Replace only if value exists
            print("Replacing placeholder: ", placeholder, "with value: ", all_vars[placeholder])
            updated_lines = updated_lines.replace(f"{{{{{placeholder}}}}}", str(all_vars[placeholder]))
    return updated_lines


def extract_variable_from_response(response, all_vars):
    matches = re.findall(r'<(.*?)>(.*?)</\1>', response)
    for match in matches:
        variable_name, variable_value = match
        if variable_value:  # Ensure value is not empty
            all_vars[variable_name.strip()] = variable_value.strip()
    return all_vars


def uncapitalize(phrase):
    return phrase[:1].lower() + phrase[1:] if phrase else phrase


def convert_number_to_float(number):

    if isinstance(number, (int, float)):
        return float(number)
    
    # Convert to string and clean up
    number = str(number).replace(',', '').strip()
    
    # Handle percentages
    if '%' in number:
        number =  number.replace('%', '')
    
    # Handle currency values
    if '$' in number:
        number = number.replace('$', '')
    
    if "million" in number:
        number = number.replace('million', '').strip()
    
    if 'cm' in number:
        number = number.replace('cm', '').strip()
    
    if 'approximately' in number:
        number = number.replace('approximately', '').strip()

    # New check: if there's a parenthesis, only use the part before it.
    if '(' in number:
        number = number.split('(')[0].strip()

    # Remove any non-numeric characters except decimal point and minus sign
    print("Number before cleaning: ", number)
    number = re.sub(r'[^\d.-]', '', number)
    print("Number after cleaning: ", number)
    if not number:
        return None
    if number == '':
        return None
    return float(number)



def distribution_plots(results, variables, experiment_spec, results_file_name):
    num_vars = len(results)
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 3 * num_vars))  # Increase figure size

    for ax, (var_name, info) in zip(axes, results.items()):
        prior_distribution = info['fitted_prior']
        prior_type = prior_distribution['type']

        if prior_type == 'gaussian':
            if prior_distribution['mean'] is None or prior_distribution['std'] is None:
                print("Gaussian parameters are None for variable: ", var_name)
                continue
            ground_truth = get_variable_mean(var_name, variables)
            mean = prior_distribution['mean']
            std = prior_distribution['std']
            x = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
            y = norm.pdf(x, mean, std)
            prior_mean = results[var_name]['processed_results']['prior_mean']
            prior_mode = results[var_name]['processed_results']['prior_mode']
            lower_quartile = norm.ppf(0.25, mean, std)
            upper_quartile = norm.ppf(0.75, mean, std)
            ax.plot(x, y, label=f'Prior – Gaussian Distribution')
        elif prior_type == 'beta':
            if prior_distribution['a'] is None or prior_distribution['b'] is None:
                print("Beta parameters are None for variable: ", var_name)
                continue
            ground_truth = get_variable_mean(var_name, variables)
            a = prior_distribution['a']
            b = prior_distribution['b']
            x = np.linspace(0, 1, 1000)
            y = beta.pdf(x, a, b)
            prior_mean = results[var_name]['processed_results']['prior_mean']
            if 'prior_mode' in results[var_name]['processed_results']:
                prior_mode = results[var_name]['processed_results']['prior_mode']
            else: 
                prior_mode = None
            lower_quartile = beta.ppf(0.25, a, b)
            upper_quartile = beta.ppf(0.75, a, b)
            ax.plot(x, y, label=f'Prior – Beta Distribution')
        else:
            raise ValueError("Unknown distribution type: ", prior_type)

        # Overlay ground truth
        ax.axvline(ground_truth, color='red', linestyle='--', linewidth=1.5, label='Ground Truth')
        ax.axvline(prior_mean, color='blue', linestyle='--', linewidth=1.5, label='Prior Mean')
        if prior_mode is not None: 
            ax.axvline(prior_mode, color='green', linestyle='--', linewidth=1.5, label='Prior Mode')
        
        # Overlay quartiles
        ax.axvline(lower_quartile, color='purple', linestyle='--', linewidth=1.5, label='Lower Quartile (25th)')
        ax.axvline(upper_quartile, color='orange', linestyle='--', linewidth=1.5, label='Upper Quartile (75th)')

        # Wrap title text and adjust its position above the plot
        wrapped_title = textwrap.fill(f'Distribution for {var_name}', width=30)
        ax.set_title(wrapped_title, fontsize=8, pad=20)  # Smaller font and more padding
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_file_name}/distribution_plots.png")


def expected_abs_distance_gaussian(mu, sigma, ground_truth, base_var_norms):
    """Computes E[|X - a|] for a normal distribution N(mu, sigma^2) with values normalized to [0,1] scale."""
    min_val = base_var_norms['min']
    max_val = base_var_norms['max']
    range_val = max_val - min_val
    
    # Normalize mu and ground_truth to [0,1] scale
    mu_norm = (mu - min_val) / range_val
    ground_truth_norm = (ground_truth - min_val) / range_val
    
    # Scale sigma to normalized space
    sigma_norm = sigma / range_val
    
    # Define normalized probability density function
    f_x = lambda x: np.abs(x - ground_truth_norm) * stats.norm.pdf(x, loc=mu_norm, scale=sigma_norm)
    
    # Integrate over [0,1] in normalized space
    result, _ = integrate.quad(f_x, 0, 1)
    return result


def expected_abs_distance_beta(alpha, beta, ground_truth):
    """Computes E[|X - a|] for a Beta(alpha, beta) distribution (defined on [0,1])."""
    f_x = lambda x: np.abs(x - ground_truth) * stats.beta.pdf(x, alpha, beta)

    # Compute integral over the support [0,1]
    result, _ = integrate.quad(f_x, 0, 1)
    return result


def get_distribution_mode(prior_distribution):
    if 'type' not in prior_distribution: 
        if 'alpha' in prior_distribution: 
            prior_type = 'beta'
            a = prior_distribution['alpha']
            b = prior_distribution['beta']
        elif 'mu' in prior_distribution: 
            prior_type = 'gaussian'
            mean = prior_distribution['mu']
        else: 
            raise ValueError("Unknown distribution type: ", prior_distribution)
    else: 
        prior_type = prior_distribution['type']
        if prior_type == 'beta': 
            a = prior_distribution['a']
            b = prior_distribution['b']
        elif prior_type == 'gaussian': 
            mean = prior_distribution['mean']
        else: 
            raise ValueError("Unknown distribution type: ", prior_distribution)
    if prior_type == 'gaussian':
        return mean 
    elif prior_type == 'beta':
        # Handle special cases
        if a < 1 and b < 1:
            return 0.5  # U-shaped, return center point between the two modes
        elif a < 1 and b > 1:
            return 0.0  # Mode at x = 0
        elif a > 1 and b < 1:
            return 1.0  # Mode at x = 1
        elif a == 1 and b == 1:
            return 0.5  # Uniform distribution, return center point
        else:  # a > 1 and b > 1
            return (a - 1) / (a + b - 2)  # Interior mode
    else:
        raise ValueError("Unknown distribution type: ", prior_type)