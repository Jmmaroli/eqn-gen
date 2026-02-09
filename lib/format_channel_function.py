"""
Format channel function strings for display and saving.

This module provides utilities for formatting channel function data into
human-readable strings showing templates, estimates, and their mappings.
"""


def format_channel_function(channel_function, channel_number):
    """
    Format a channel function into template, estimate, and mapping strings.
    
    Args:
        channel_function: List of product function dictionaries, each containing:
            - "template_string": String representing the function template
            - "estimate_string": String representing the estimated function (or None)
        channel_number: Output channel number
    
    Returns:
        tuple: Three strings:
            1. y_str_template: Channel function template equation (e.g., "y1[k] = f1(x1) + f2(x2)")
            2. y_str_estimate: Channel function estimate equation (e.g., "y1[k] = 1.5*x1 + 2.3*x2")
            3. y_str_mappings: Template-to-estimate mappings (e.g., "f1(x1) = 1.5*x1\nf2(x2) = 2.3*x2")
    
    Example:
        >>> channel_function = [
        ...     {"template_string": "f1(x1)", "estimate_string": "1.5*x1"},
        ...     {"template_string": "f2(x2)", "estimate_string": "2.3*x2"}
        ... ]
        >>> template, estimate, mappings = format_channel_function(channel_function, 0)
        >>> print(template)
        y1[k] = f1(x1) + f2(x2)
        >>> print(estimate)
        y1[k] = 1.5*x1 + 2.3*x2
        >>> print(mappings)
        f1(x1) = 1.5*x1
        f2(x2) = 2.3*x2
    """
    # Initialize strings for equations
    y_str_template = "y" + str(channel_number) + "[k] = "
    y_str_estimate = "y" + str(channel_number) + "[k] = "

    # Build the equations term by term
    for idf, product_function in enumerate(channel_function):
        template = product_function["template_string"]
        estimate = product_function["estimate_string"] if product_function["estimate_string"] is not None else product_function["template_string"]

        # Append to the main template equation
        y_str_template += template
        # Append to the main estimate equation
        y_str_estimate += estimate

        # Add " + " if not the last term
        if idf < len(channel_function) - 1:
            y_str_template += " + "
            y_str_estimate += " + "

    # Build template-to-estimate mappings
    mapping_lines = []
    for product_function in channel_function:
        template = product_function["template_string"]
        estimate = product_function["estimate_string"] if product_function["estimate_string"] is not None else product_function["template_string"]
        mapping_lines.append(f"{template} = {estimate}")
    
    y_str_mappings = "\n".join(mapping_lines)

    return y_str_template, y_str_estimate, y_str_mappings
