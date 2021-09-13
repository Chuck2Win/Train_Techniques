def train(model, optimizer, criterion, epoch, train_dataloader, val_dataloader, logger,ear):
    # BERT
    Loss = []
    for epoch in tqdm(range(1,epoch+1)):
        model.train()
        Loss_t = 0.
        for data in train_dataloader:#tqdm(train_dataloader):
            optimizer.zero_grad()
            data = (i.to(args['device']) for i in data)
            input_ids, attention_masks, token_type_ids, labels = data
            output = model.forward(input_ids, attention_masks, token_type_ids)
            loss = criterion(output, labels)
            Loss_t+=loss.item()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        Loss.append(Loss_t/len(train_dataloader))
        logger.info('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))

        print('epoch : %d ----- Train_Loss : %.4f'%(epoch+1,Loss[-1]))
        model.eval()
        with torch.no_grad():
            actual = []
            pred = []
            Loss_val = 0.
            for data in val_dataloader:
                data = (i.to(args['device']) for i in data)
                input_ids, attention_masks, token_type_ids, labels = data
                output = model.forward(input_ids, attention_masks, token_type_ids)
                loss = criterion(output, labels)
                Loss_val+=loss
                actual.extend(labels.tolist())
                pred.extend(output.argmax(-1).tolist())
            Loss_val = Loss_val/len(val_dataloader)
            logger.info('epoch : %d ----- Val_Loss : %.4f'%(epoch,Loss_val))
            logger.info(classification_report(actual,pred))
            logger.info('='*100)
            print('epoch : %d ----- Val_Loss : %.4f'%(epoch,Loss_val))
            print(classification_report(actual,pred))
            print('='*100)
        ear.check(model,Loss_val)
        if ear.timetobreak:
            logger.info('epoch : %d ----- Train END'%(epoch))
            print('epoch : %d ----- Train END'%(epoch))    
            return Loss
    logger.info('train end')
    torch.save(model, ear.save_dir)
    return Loss    
