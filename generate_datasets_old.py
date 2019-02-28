def generate_datasets_2(operand_digits, operator):
    if operator == 'add':
        op_dataset, carry_datasets = generate_add_datasets(operand_digits)
    if operator == 'subtract':
        op_dataset, carry_datasets = generate_subtract_datasets(operand_digits)
    if operator == 'multiply':
        op_dataset, carry_datasets = generate_multiply_datasets(operand_digits)
    if operator == 'divide':
        op_dataset, carry_datasets = generate_divide_datasets(operand_digits)
    if operator == 'modulo':
        op_dataset, carry_datasets = generate_modulo_datasets(operand_digits)

    return op_dataset, carry_datasets


def generate_add_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries addition.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries addition.
    -- result_digits == operand_digits + 1

    '''
    op_dataset = {'input':list(), 'output':list()}
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of an adding operation
            result, n_carries = add_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of addition.
            input = np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1)
            op_dataset['input'].append(input)
            carry_datasets[n_carries]['input'].append(np.concatenate(input))

            # Append the output of addition.
            output = result.reshape(1,-1)
            op_dataset['output'].append(output)
            carry_datasets[n_carries]['output'].append(output)

    # List to one numpy.ndarray
    op_dataset['input'] = np.concatenate(op_dataset['input'], axis=0)
    op_dataset['output'] = np.concatenate(op_dataset['output'], axis=0)

    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    # Shuffle the pairs of input and output of op_dataset.
    op_dataset['input'], op_dataset['output'] = shuffle_io_pairs(op_dataset['input'], op_dataset['output'])

    return op_dataset, carry_datasets


def generate_subtract_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries subtraction.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries subtraction.
    -- result_digits == operand_digits
    '''
    op_dataset = {'input':list(), 'output':list()}
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            if dec_op1 >= dec_op2:
                # Get numpy.ndarray binary operands.
                np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
                np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

                # The phase of a subtracting operation
                result, n_carries = subtract_two_numbers(np_bin_op1, np_bin_op2)

                # Create a list to store operations
                if n_carries not in carry_datasets:
                    carry_datasets[n_carries] = dict()
                    carry_datasets[n_carries]['input'] = list()
                    carry_datasets[n_carries]['output'] = list()

                # Append the input of subtraction.
                input = np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1)
                op_dataset['input'].append(input)
                carry_datasets[n_carries]['input'].append(input)

                # Append the output of subtraction.
                output = result.reshape(1,-1)
                op_dataset['output'].append(output)
                carry_datasets[n_carries]['output'].append(output)

    # List to one numpy.ndarray
    op_dataset['input'] = np.concatenate(op_dataset['input'], axis=0)
    op_dataset['output'] = np.concatenate(op_dataset['output'], axis=0)

    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    # Shuffle the pairs of input and output of op_dataset.
    op_dataset['input'], op_dataset['output'] = shuffle_io_pairs(op_dataset['input'], op_dataset['output'])

    return op_dataset, carry_datasets


def generate_multiply_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries multiplication.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries multiplication.
    -- result_digits == operand_digits * 2
    '''
    op_dataset = {'input':list(), 'output':list()}
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(2**operand_digits):
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a multiplying operation
            result, n_carries = multiply_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of multiplication.
            input = np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1)
            op_dataset['input'].append(input)
            carry_datasets[n_carries]['input'].append(input)

            # Append the output of multiplication.
            output = result.reshape(1,-1)
            op_dataset['output'].append(output)
            carry_datasets[n_carries]['output'].append(output)

    # List to one numpy.ndarray
    op_dataset['input'] = np.concatenate(op_dataset['input'], axis=0)
    op_dataset['output'] = np.concatenate(op_dataset['output'], axis=0)

    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    # Shuffle the pairs of input and output of op_dataset.
    op_dataset['input'], op_dataset['output'] = shuffle_io_pairs(op_dataset['input'], op_dataset['output'])

    return op_dataset, carry_datasets


def generate_divide_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries division.
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries division.
    -- result_digits == operand_digits
    '''
    op_dataset = {'input':list(), 'output':list()}
    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(1, 2**operand_digits): # Exclude `dec_op2 = 0`
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a dividing operation
            result, n_carries, _ = divide_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of division.
            input = np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1)
            op_dataset['input'].append(input)
            carry_datasets[n_carries]['input'].append(input)

            # Append the output of division.
            output = result.reshape(1,-1)
            op_dataset['output'].append(output)
            carry_datasets[n_carries]['output'].append(output)

    # List to one numpy.ndarray
    op_dataset['input'] = np.concatenate(op_dataset['input'], axis=0)
    op_dataset['output'] = np.concatenate(op_dataset['output'], axis=0)

    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    # Shuffle the pairs of input and output of op_dataset.
    op_dataset['input'], op_dataset['output'] = shuffle_io_pairs(op_dataset['input'], op_dataset['output'])

    return op_dataset, carry_datasets


def generate_modulo_datasets(operand_digits):
    '''
    Parameters
    ----------
    operand_digits: the number of the digits of an operand

    Returns
    -------
    carry_datasets: dict.
    - carry_datasets[n_carries]['input']: numpy.ndarray. shape == (n_operations, operand_digits * 2).
    -- Input dataset for n_carries modulo(division).
    - carry_datasets[n_carries]['output']: numpy.ndarray. shape == (n_operations, result_digits).
    -- Output dataset for n_carries modulo(division).
    -- result_digits == operand_digits
    '''

    carry_datasets = dict()
    for dec_op1 in range(2**operand_digits):
        for dec_op2 in range(1, 2**operand_digits): # Exclude `dec_op2 = 0`
            # Get numpy.ndarray binary operands.
            np_bin_op1 = get_np_bin(get_str_bin(dec_op1), operand_digits)
            np_bin_op2 = get_np_bin(get_str_bin(dec_op2), operand_digits)

            # The phase of a dividing operation
            result, n_carries = modulo_two_numbers(np_bin_op1, np_bin_op2)

            # Create a list to store operations
            if n_carries not in carry_datasets:
                carry_datasets[n_carries] = dict()
                carry_datasets[n_carries]['input'] = list()
                carry_datasets[n_carries]['output'] = list()

            # Append the input of division.
            carry_datasets[n_carries]['input'].append(np.concatenate((np_bin_op1, np_bin_op2)).reshape(1,-1))

            # Append the output of division.
            carry_datasets[n_carries]['output'].append(result.reshape(1,-1))

    # List to one numpy.ndarray
    for key in carry_datasets.keys():
        carry_datasets[key]['input'] = np.concatenate(carry_datasets[key]['input'], axis=0)
        carry_datasets[key]['output'] = np.concatenate(carry_datasets[key]['output'], axis=0)

    return carry_datasets
