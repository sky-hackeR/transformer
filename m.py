def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # Convert the tensor to a list of token IDs
            model_out_ids = model_out.squeeze(0).tolist()
            model_out_text = tokenizer_tgt.decode(model_out_ids)

            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Predicted: {model_out_text}")

            if count == num_examples:
                break
