//
//  ViewController.m
//  BananaCoreML
//
//  Created by kuraki on 2023/12/7.
//

#import "ViewController.h"
#import "BananaCoreML-Swift.h"
#import <AVFoundation/AVFoundation.h>

@interface ViewController()

@property (nonatomic, strong) NSTextField *textField;

@property (nonatomic, strong) BananaModel *model;

@end

@implementation ViewController

- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupUI];
}

- (void)setupUI
{
    NSButton *button1 = [NSButton buttonWithTitle:@"选择训练集" target:self action:@selector(trainModel)];
    button1.frame = CGRectMake(150, 0, 100, 100);
    [self.view addSubview:button1];
    
    NSButton *button2 = [NSButton buttonWithTitle:@"选择要预测成熟度的香蕉图" target:self action:@selector(predictRipe)];
    button1.frame = CGRectMake(0, 0, 100, 100);
    [self.view addSubview:button2];
    
    _textField = [[NSTextField alloc] initWithFrame:CGRectMake(0, 150, 1000, 500)];
    [self.view addSubview:_textField];
}

- (void)trainModel
{
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;

    [panel beginWithCompletionHandler:^(NSInteger result){
        if (result == NSFileHandlingPanelOKButton) {
            NSURL*  theDoc = [[panel URLs] objectAtIndex:0];
            [self.model trainBananaRipeWithUrl:theDoc completionHandler:^(NSString * _Nullable trainMessage) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.textField.stringValue = trainMessage;
                });
            }];
        }
    }];
}

- (void)predictRipe
{
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;

    [panel beginWithCompletionHandler:^(NSInteger result){
        if (result == NSFileHandlingPanelOKButton) {
            NSURL*  theDoc = [[panel URLs] objectAtIndex:0];
            NSImage *image = [[NSImage alloc] initWithContentsOfURL:theDoc];
            CGImageSourceRef source;
            source = CGImageSourceCreateWithData((CFDataRef)[image TIFFRepresentation], NULL);
            CGImageRef maskRef =  CGImageSourceCreateImageAtIndex(source, 0, NULL);
            [self.model predictBananaRipeWithImage:[CIImage imageWithCGImage:maskRef] completionHandler:^(float result) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    NSString *trainMsg = self.textField.stringValue;
                    self.textField.stringValue = [NSString stringWithFormat:@"%@\n\n预测香蕉成熟度:%f",trainMsg,result];
                });
            }];
        }
        
    }];
}


- (BananaModel *)model
{
    if (!_model) {
        _model = [[BananaModel alloc] init];
    }
    return _model;
}


@end
